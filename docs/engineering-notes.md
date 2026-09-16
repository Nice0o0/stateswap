# 工程踩坑笔记（Windows + RTX 5070 Ti / Blackwell 复现 RWKV-7 state tuning）

> 这份笔记记录本项目在 Consumer Blackwell GPU + 原生 Windows 上搭建
> RWKV-7 state tuning 训练/服务栈时踩过的每一个坑。每一条都是真实发生、
> 有日志佐证的，供后来者参考。

## 1. 网络：GFW 环境下的依赖与模型获取

- `pypi.org`、`download.pytorch.org`（索引页）可直连，但 torch 轮子的真实
  下载域 `download-r2.pytorch.org` TLS 握手被掐。
- `hf-mirror.com`、`github.com`、`ghproxy.net` 直连均失败（DNS 污染 / TLS 重置）。
- 本机 Clash 系代理（127.0.0.1:7890）对不同域名行为不一致：
  `download-r2.pytorch.org` 走代理返回 200，`github.com` 走代理 TLS EOF。
- 最终方案：
  - **torch**：`curl -x 代理 --retry 20 -C -` 断点续传 2.75GB 轮子
    （uv/pip 走代理反而握手失败，curl 能通——同一代理不同客户端行为不同）。
  - **RWKV-7 权重**：ModelScope 官方镜像 `RWKV/rwkv-7-world`，国内直连满速。
  - **fla 转换器 / 词表 / NekoQA 数据**：jsDelivr CDN 代理 GitHub raw。

## 2. Python 3.10 + triton-windows：`inspect.getsourcelines` 截断 bug

- triton-windows 3.8.0.post28 提供 cp310 轮子，导入 fla 时在
  `triton/runtime/jit.py` 的 `get_def_col_number` 抛
  `ValueError: No function definition found for kernel`。
- monkeypatch 抓现场：`inspect.getsourcelines(fn)` 对"多层装饰器"（外层
  `@triton.heuristics({...})` + 内层 `@triton.jit`）只返回到第一个装饰器
  闭括号为止的 **258 个字符**，`def` 行和函数体全部丢失——
  CPython 3.10 的 tokenize/inspect 行为与 3.11+ 不同。
- 结论：不跟 CPython 内部机制搏斗，**直接换 Python 3.12**。
- 副产品：uv 下载 managed Python 需要 GitHub，直连不通；用
  `UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone`
  从 npmmirror 拿，顺利建 3.12 环境。

## 3. Blackwell (sm_120) 上 Triton 内核的真实状态

- RTX 5070 Ti（ Laptop，12GB，capability (12,0)）+ torch 2.11.0+cu128
  + triton-windows 3.8.0.post28：fla 0.5.2 的全部 RWKV7 相关内核
  （chunk、fused_recurrent、token_shift、fused_addcmul、l2norm、
  layernorm、gate_output_correction）**编译、autotune、运行全部成功**。
- kernel 冒烟必须用真实 dtype 组合：`r/w/k/v/a/b` 全 bf16、state fp32；
  我最初把 `w` 传成 fp32 直接触发
  `Both operands must be same dtype. Got bf16 and fp32`。

## 4. fla 0.5.2 的两个真实 bug（本项目发现并绕过）

### 4.1 `fused_mul_recurrent` 路径的 initial_state 梯度丢失

- 现象：`model.eval()` 且序列 < 64 token 时层走 `fused_mul_recurrent_rwkv7`，
  `loss.backward()` 报 "does not require grad" —— S0 的梯度链整条断裂。
- 验证方法：同样的张量在 chunk 路径梯度正常 → 问题锁定在
  fused_recurrent 路径的 autograd 实现。
- 绕过：训练时强制 `model.train()`（fla 层用 `self.training or seq_len >= 64`
  选 chunk 路径）；RWKV7 无 dropout/BN，train 模式无副作用。
- 这正是 Preen 文档警告的"梯度静默断裂"——不报错、只是训不动，
  必须靠"梯度范数非零且有限"的冒烟测试来抓。

### 4.2 RWKV7Config 的 `num_heads` 陷阱

- `RWKV7Config.__init__` 用**默认** hidden_size 预算 `num_heads`；
  转换器事后覆盖 `config.hidden_size` 不会触发重算。
- 模型本身不受影响（`head_dim` 分支优先，`num_heads = hidden // head_dim`），
  但任何相信 `cfg.num_heads` 的下游代码（比如 S0 的形状）会拿到错值：
  0.1B 实际 12 头，config 里却是 32。
- 修复：转换器里显式 `config.num_heads = hidden_size // head_dim`，
  并在 S0 里不信任 config、自行计算。

## 5. 训练稳定性：NaN 与巨型梯度

- 0.4B、lr=1e-4、batch=1：800 步训练在 100~200 步之间出现 NaN
  （GPU 非确定性，两次同 seed 运行结果不同）。
- 防护三件套：
  1. loss 非有限 → 跳过该 batch，不进 Adam（一个 NaN batch 的动量
     会永久污染优化器）；
  2. 梯度范数非有限 → 跳过更新；
  3. `clip_grad_norm_(1.0)`。
- 实测梯度范数本身巨大（1e2 ~ 4e3），被 clip 归一后训练稳定；
  |S0|∞ 全程 < 0.01，无 state 爆炸（Preen 记录的 lr=1.0 爆炸在 1e-4 下不会发生）。
- 仪表：每 50 步打印 `|g|`、`|S0|∞`、峰值显存、s/step——
  这些曲线是判断"训练是否健康"的第一手证据。

## 6. 其他

- **位置参数在签名扩张后静默错位**（本轮最疼的坑）：`chat_stream` 在
  `top_p` 与 `no_repeat_ngram` 之间隔着 `rep_penalty`，给 engine 加
  n-gram 参数时 server 调用点按位置传参没改全，`no_repeat_ngram=8` 落进了
  `rep_penalty`——此后每条 API 请求都以 8 倍重复惩罚采样，长回复被强制
  避开近期词而胡言乱语。教训：**跨层调用一律用关键字参数**，签名扩张后
  grep 全部调用点。
- **长对话雪崩与退化护栏**：RWKV 的递归状态就是对话记忆本身——长回复
  一旦中途脱轨，脱轨 token 也被写进状态，后续轮次从被污染的状态继续，
  越滚越糟（probe_longconv.py：T11 长故事退化 → T12–15 全程复读）。
  注意这与"状态相对 S₀ 漂移"无关：实测 cos(状态, S₀) 两轮内跌到 0.05
  以下而人格保持完好，S₀ 只负责偏置轨迹起点。护栏 = 轮初快照 +
  字符 4-gram 唯一率检测 + 回滚（engine.py）。**朴素 S₀ 回锚
  （S ← 0.85S + 0.15S₀）反而制造泰文汤**——运行态状态的线性插值同样
  在流形之外，与 state 算术的相变结论一致（probe_persona_decay.py）。
- **多轮样本训练是治本项**：单轮样本只在"对话开头"训练 S0，长对话中段的
  状态分布从未被覆盖。`--turns 3 --ctx 1024` 把若干对拼成多轮对话
  （每个 Assistant 段都算 loss），梯度穿过状态演化后的每一轮——
  同样 15 轮探针退化 4/15 → 0/15（benchmarks.md §7）。话题不连贯不要紧，
  训练的正是"状态演化后仍保持人格"的稳健性。训练人格用于长对话场景时
  默认应开 --turns。
- **递归状态保不住具体事实**：6 轮回忆探针（probe_context.py）：埋点
  "我叫小李/学吉他/狗叫豆豆"，neko 人格即刻回忆即幻觉（"你叫小鱼干"），
  none 基线直接死循环。上下文影响只剩"话题词渗透"。根因有二：训练数据
  没有跨轮依赖（组链是独立单轮对拼接，模型无处学"记住"）；1.5B 的
  64×64 层状态对稀疏事实的保持本就脆弱。工程折中：**有界文本重放**
  （`context_replay`，默认 4）——近期事实以可见文本进 prompt，
  K=2 覆盖 2 轮内、K=4 覆盖 4 轮内；更早的事实与代词回指是 1.5B 的
  硬边界。注意重放只影响 prefill 文本，O(1) 状态语义不变。
- transformers 5.16 的 tied-weight 记账与 fla 的 `_tied_weights_keys`
  （list 形式）不兼容，`save_pretrained` 直接崩；绕过：手动写
  safetensors + `config.save_pretrained`。
- RWKV World 词表（`rwkv_vocab_v20230424.txt`）行格式是
  `<id> '<python repr token>' <freq>`，token 可能含空格和两种引号——
  必须取"行内第一个引号到最后一个引号"，再按
  `ord(c) < 256 ? 单字节 : utf-8(c)` 还原字节流。
- 自写 tokenizer 用字节 trie + 贪婪最长匹配，与 BlinkDL 参考实现语义一致；
  解码用增量 UTF-8 decoder 避免多字节字符被 token 边界切碎。

## 复现环境快照

- Windows 10 (19045)，RTX 5070 Ti Laptop 12GB，驱动 581.80（CUDA 13.0）
- Python 3.12.13（uv managed，npmmirror 源）
- torch 2.11.0+cu128 / triton-windows 3.8.0.post28 / flash-linear-attention 0.5.2
  / transformers 5.16.1 / fastapi 0.141
