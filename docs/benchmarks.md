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

同一引擎、同一模型的两种服务架构对比：

| 历史长度 | 无状态 prefill（每请求重放历史） | session 模式（O(1) 状态续聊） |
|---|---|---|
| ~500 tok | 65.5 ms | ~50 ms |
| ~1000 tok | 73.2 ms | ~50 ms |
| ~2000 tok | 75.5 ms | ~50 ms |
| ~4000 tok | 70.3 ms | ~50 ms |

session 模式的 prefill 恒定 ~50ms（12 轮对话无增长）；无状态模式的开销
随历史线性增长（chunk 内核在小规模下并行度高，差距在本硬件上不大，
但会话状态内存恒定 + 无需重传历史是架构级优势；Transformer 同场景
还需随历史增长的 KV cache 内存）。

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

### 6.2 S₀ int8 量化（`src/stateswap/quant.py`）

逐 (layer, head) 对称 int8：**12.58 MB → 3.15 MB（4.0×）**。
贪心对比 6 条 prompt：1 条逐字节相同，5 条前半句相同、后段因量化噪声分叉——
**风格与语义完全保真，精确 token 路径不保证**（量化噪声随生成放大，符合预期）。

### 6.3 torch.compile / CUDA Graph（阴性结果）

`torch.compile(mode="reduce-overhead")` 反而 **0.77×**（17.8 vs 23.0 tok/s）。
原因：fla Cache 逐 token 的 python dict 更新 + triton 内核变异识别失败
（"assuming every input is mutated"）导致大量图断裂。要吃到 CUDA Graph
收益需要手动静态缓冲区改造（状态张量固定 + copy_ 搬运），是独立的工程项，
不是一个开关。

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
