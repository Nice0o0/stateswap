# Benchmarks

硬件：RTX 5070 Ti Laptop 12GB（Blackwell sm_120），原生 Windows 10，
torch 2.11.0+cu128，triton-windows 3.8.0.post28，fla 0.5.2，
模型 RWKV7-World-0.4B（bf16 冻结），单进程单请求。

原始数据见 `docs/benchmarks.json`（由 `python -m stateswap.bench` 生成）。

## 1. S0 训练（权重全冻结，仅训初始状态）

| 项 | 数值 |
|---|---|
| 0.4B / ctx 512 / batch 1 | **0.23–0.33 s/step**，峰值显存 1.5 GB |
| 0.1B / ctx 256 / batch 1 | ~0.27 s/step，峰值显存 0.8 GB |
| S0 参数量 | 0.4B：1.57M（6.29 MB fp32）；0.1B：0.59M（2.36 MB） |
| NekoQA-200 / 800 步 | loss 3.80 → 2.31 |
| zh2en-50 / 400 步 | loss 1.50 → 0.018（训练分布内近乎完全掌握） |

## 2. 行为验证

- **风格命中率**（回复含 猫娘标记词/动作括号）：
  - neko-0.4b（200 对语料 / 800 步）：基线 **0–12%** → **88–100%**（温度 1.0，两次运行）
  - neko-0.4b-v2（3,095 对语料 / 4000 步）：基线 **25%** → **100%**（贪心解码，
    8/8 全部在角色内；原始对比见 `docs/benchmarks_v2.json`）
- **无指令自主翻译**（zh2en-0.4b-v3，WMT 7,369 对 + 50 口语对，裸格式）：
  贪心解码下 8 个未见于训练集的裸中文句 **100% 输出纯英文**。
  注意：50 对裸句训练的 zh2en-v2 只能记忆训练句、零泛化（见
  [state-arithmetic.md](state-arithmetic.md) §3.4）——任务不变性需要数据规模。

## 3. Serving（engine.py + FastAPI）

| 项 | 数值 |
|---|---|
| 人格热切换（重建 24 层状态缓存） | **2.6–4.4 ms** |
| 每会话状态内存（24 层 recurrent + conv/ffn） | **6.39 MB，恒定不随轮数增长** |
| 人格文件体积 | 6.29 MB（fp32，可压缩至 bf16/int8） |
| 解码速度（0.4B，纯 Python 循环） | ~20–21 tok/s（47–50 ms/token） |
| 预填充（~22–34 token 输入） | 48–75 ms |

## 4. Session 模式 vs 无状态模式

同一引擎、同一模型的两种服务架构对比。**1.5B，稳态（预热后）数字**：

| 历史长度 | 无状态 prefill（每请求重放历史） | session 模式（O(1) 状态续聊） |
|---|---|---|
| 493 tok | 62.8 ms | ~50 ms |
| 1015 tok | 60.0 ms | ~50 ms |
| 2030 tok | 60.2 ms | ~50 ms |
| 4089 tok | 56.5 ms | ~50 ms |

**修正说明（2026-11）**：本表此前的 1.5B 数字（6449 / 6434 / 10304 ms）以及
0.4B 的那一版，都是在**未预热**的情况下测的，因而把 Triton autotune 的
首次编译开销算进了 prefill。同一长度首次调用与稳态可以差 **30–57 倍**：

| tokens | 首次（含 autotune） | 稳态 | 倍数 |
|---|---|---|---|
| 489 | 3374 ms | 60 ms | 56.6× |
| 1449 | 3485 ms | 121 ms | 28.8× |
| 2409 | 10938 ms | 197 ms | 55.5× |
| 5769 | 14482 ms | 493 ms | 29.4× |

原始对照见 `docs/prefill-cold-vs-warm.json`。`bench_prefill_only` 已改为
每长度跑两次、只取第二次，并记录 `first_call_ms` 备查。

**0.4B 不受影响**：重测（`docs/prefill-0.4b-warm.json`）得
68.7 / 77.2 / 70.4 / 73.6 ms，与旧值 65.5 / 73.2 / 75.5 / 70.3 基本一致——
该档位的内核 config 少，autotune 只在首次调用时发生（3,000 ms 级），没进旧表。
所以受污染的只有 1.5B 那份。

**修正后的结论比原来更强**：稳态 prefill 从 493 到 4089 token 只从 62.8 变到
56.5 ms——**不随历史增长**（chunk 内核在这个尺度下并行度足够高，预处理开销
被摊平）。所以 session 模式的价值不在「省 prefill 时间」（两者都约 60 ms），
而在于 **(a) 不用重传历史、(b) 内存恒定 12.78 MB 而非随历史增长、
(c) 会话状态可持久化/热切换**——这正是本项目的架构论点，之前的错误数据
反而掩盖了它。

## 5. 诚实性声明

- decode ~20 tok/s 是纯 Python 逐 token 循环 + CUDA launch 开销的结果，
  未做 CUDA Graph / C++ runtime 优化（见 Roadmap）。
- style 命中率是启发式指标（标记词匹配），不是人工评审。
- 无状态对比项在"同一 RWKV 引擎"上进行，不是对 Transformer
  serving 栈（vLLM 等）的直接对比。
- 基准均为单次运行、单请求；未测并发吞吐。

## 6. 优化实验（Roadmap 三项实测）

### 6.1 多会话批量解码（batch_chat）

把 B 个会话的递归状态沿 batch 维堆叠、一次前向推进（`engine.batch_chat`）：

| B | 逐会话 tok/s | 批量 tok/s | 加速比 |
|---|---|---|---|
| 1 | 14.1 | 22.4 | 1.59× |
| 2 | 20.6 | 33.8 | 1.64× |
| 4 | 21.6 | 64.2 | 2.97× |
| 8 | 21.5 | **135.5** | **6.29×** |

**8 会话并发近线性扩展（6.29×）**，证实"RWKV 无 attention、batch 成本近似线性"。
单会话逐循环上界 ~21 tok/s 的瓶颈在 Python 每步开销；批量把 B 份 Python 开销摊薄。

### 6.1b 采样器热点：重复惩罚向量化

逐 token 剖析（0.4B，非贪心路径 `T=0.8 / top_p=0.8 / rep=1.25 / ngram=8`）：

| 环节 | 耗时 |
|---|---|
| 模型前向（单 token，bf16） | **47 ms**（占绝对主导） |
| `_sample` 整体 | 5.8 ms |
| ↳ 重复惩罚（旧：逐 token 标量循环，≤128 次） | **6.05 ms** |
| ↳ 重复惩罚（新：一次 gather+where+scatter） | **0.12 ms** |
| ↳ softmax(65536) | 0.015 ms |
| ↳ argsort(65536) | 0.12 ms |
| ↳ n-gram 封禁集合（纯 Python） | 0.02 ms |

旧实现是解码路径里唯一"随历史长度增长的 GPU 往返"：`.item()`/标量写回
逐个 launch，128 个已见 token = 128 次 kernel。向量化后数值完全等价
（`torch.where(v < 0, v, v / rep)` 与逐项 `logit *= 1 or 1/rep` 对齐到
float32 舍入），实测 6.05 → 0.12 ms（50×）。端到端非贪心解码从
~54 ms/tok 降到 ~48 ms/tok，与贪心路径（47–48 ms/tok）持平——
**采样器不再是可测瓶颈，剩下的全是模型前向**。

结论：这个路径的下一步只有 CUDA Graph / 自定义 kernel（见 §6.3 的阴性结果
与 Albatross），纯 Python 侧已无收益空间。

### 6.2 S₀ int8 量化（`src/stateswap/quant.py`）

逐 (layer, head) 对称 int8：**12.58 MB → 3.15 MB（4.0×）**。
贪心对比 6 条 prompt：1 条逐字节相同，5 条前半句相同、后段因量化噪声分叉——
**风格与语义完全保真，精确 token 路径不保证**（量化噪声随生成放大，符合预期）。

### 6.3 torch.compile / CUDA Graph（阴性结果，两种实现都测过）

**A. `torch.compile(mode="reduce-overhead")` — 0.77×**（17.8 vs 23.0 tok/s）。
原因：fla Cache 逐 token 的 python dict 更新 + triton 内核变异识别失败
（"assuming every input is mutated"）导致大量图断裂。

**B. 手工静态缓冲区 + `torch.cuda.CUDAGraph`（本次新增，结论：不可用）。**
先测得一个非常诱人的数字：把解码步捕获成 graph 后 **43.65 → 6.02 ms/token
（7.25×）**。但正确性检查推翻了它：

| 检查 | 结果 |
|---|---|
| capture 前预热 15 步（否则 cuBLAS handle 未初始化，capture 直接失败） | 通过 |
| replay 后 logits 变化 | ✅ |
| replay 后 `Cache` 里状态张量的**引用**是否换新 | ❌ 不变 |
| replay 后状态**数值**第一次变化 | ✅ |
| 第二次 replay 后状态继续变化 | ❌ **不再变化** |
| 24 步贪心解码 vs eager 同起点 | ❌ graph 输出 24 个相同 token（`'s's's's…`），eager 正常 |

根因：fla 的 `FLALayer.update` 把新状态**写进捕获时绑定的那条张量**（原地
`copy_` 语义由 Triton kernel 以指针实现），graph replay 每次都写回同一个
捕获期目标，而不是"上一轮输出 → 下一轮输入"的递推链。所以 7.25× 测的是
**重复写同一状态**的吞吐，不是解码吞吐。这正是 §6.3-A 预言的那件事——
"状态张量固定 + copy_ 搬运"听起来可行，但 fla 的 Cache 语义（每层 dict +
Triton 指针别名）让"固定缓冲区"变成"固定错误的缓冲区"。

**要真吃到这份收益**，需要绕开 fla Cache：自己维护每层状态张量的双缓冲，
在 graph 外做 `dst.copy_(src)` 并让 graph 只捕获"读 A 写 B"的单步，再交换。
这是独立的工程改造（估 1–2 天 + 回归全链路行为），不属于"一个开关"；
在它完成前，本项目的 decode 速度数字应视为 **CPU dispatch 受限**——
剖析显示 47 ms/token 里 GPU 实际计算 **< 0.1 ms**（`cuda_events`），
其余全是 2910 次 Triton autotuner 调用（44 us/次）与 5025 次
`torch._C._nn.linear`（30 us/次）的 Python 开销；batch=16 时每序列耗时
只降到 1/16，也印证瓶颈不在算力。**同一解释也说明为什么 0.1B 与 1.5B
的每层耗时几乎相同（2.31 vs 2.32 ms/layer）**——参数量差 15 倍，
但每层固定 launch 次数相同。

## 7. 长对话退化：机制与两层修复（15 轮探针）

场景：neko 人格连聊 15 轮（含两条"写 400 字故事"的长回复请求），
temp 0.7 / top_p 0.8 / rep 1.25 / max_new 400，退化判定 = 回复 ≥200 字符
且字符 4-gram 唯一率 < 0.75。脚本：`scripts/probe_longconv.py` /
`probe_mt.py`（guard 关闭测裸稳健性）/`probe_guard2.py`（护栏效果）。

| 配置 | 退化轮数 | 说明 |
|---|---|---|
| 单轮训练人格（neko-1.5b） | **4/15**（T11–T14 连发） | T11 长故事脱轨 → 写入状态 → T12+ 雪崩 |
| + 退化护栏（engine 默认开启） | 0 残留 | T11 起点当场回滚，T12 立即恢复；连续退化逐级加深 |
| 多轮训练人格（neko-1.5b-mt，turns=3 ctx=1024） | **0/15** | 无需护栏：uq 全程 ≥0.94，回复长度自然收敛 |

机制要点（反直觉）：

- 退化与"状态相对 S₀ 漂移"**无关**：cos(状态, S₀) 两轮内 <0.05 而人格完好，
  S₀ 只负责偏置轨迹起点。
- **朴素 S₀ 回锚有害**：S ← 0.85S + 0.15S₀ 每轮执行，T07 起产出泰文/颜文字汤
  ——运行态状态的线性插值在流形之外，与 state 算术相变结论一致
  （`scripts/probe_persona_decay.py`）。
- 多轮训练成本：1,048 条三轮链（3,095 对装箱），2000 步 ≈ 10.5 分钟
  （0.31 s/step，峰值显存 6.4GB），final loss 1.95（与单轮训练的 1.83 相当）。

护栏阈值标定：干净回复 uq4 ≥ 0.82，退化 ≤ 0.72 → 阈值 0.75。
已知盲区：字符多样的"颜文字汤"（非复读型乱码）可绕过 4-gram 检测，
宁可漏检不误杀。
