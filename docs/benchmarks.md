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

- **风格命中率**（回复含 猫娘标记词/动作括号）：基线 S0=0 约 **0–12%**，
  训练后 **88–100%**（两次运行，采样温度 1.0）。
- **无指令自主翻译**：zh2en 人格收到裸中文（无 "翻译" 指令）时直接输出英文
  ——任务模式被烘进 S0，这是 state tuning 区别于 few-shot prompt 的关键证据：

```
User: 今天的月亮又圆又亮。
[none   ] （模型自由发挥，不翻译）
[zh2en-0.4b] The moon is round and bright.
```

- 局限（如实记录）：NekoQA 只用了 200 条样本，0.4B 人格在分布外话题上
  风格保持、内容连贯性有限；上全量 10k 数据可期改善。

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
