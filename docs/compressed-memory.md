# S₀ 压缩记忆实验：知识注入的容量边界

> 研究问题：**能不能把文档知识"训"进 S₀，用 12MB 的状态替代 RAG？**
> 答案：不能（准确率 ≤13%，且不随知识量增长），但实验过程中发现并修复了
> 一个影响所有 state tuning 的分词边界 bug——两个结果都有独立价值。
>
> 数据：`docs/compressed_memory.json`、`docs/compressed_memory_control.json`；
> 脚本：`scripts/gen_knowledge_corpus.py`、`scripts/eval_memory.py`、
> `scripts/eval_memory_control.py`

## 1. 实验设计

- **受控合成知识库**：虚构"星辰科技"世界观，程序化生成 (实体, 属性, 值)
  三元组——精确 ground truth、可控知识量、可分离"记住"与"泛化"。
- **容量曲线三档**：100 / 300 / 500 条事实（1.5B 底座，S₀ 12.58MB 不变），
  训练 3000–4000 步（33 / 11 / 9 epoch，mem-100 训练 loss 归零 = 完全记住）。
- **三组探针**（贪心解码、独立会话）：
  - memorization：训练内实体 + 换一种问法（60 题）
  - generalization：完全未训的实体（40 题，测 schema 泛化）
  - general：通用能力保持探针（遗忘检查）
- **基线**：S₀=0 裸奔；RAG-oracle（金标准事实直接放进 prompt）。

## 2. 结果

| 配置 | 记忆保持 | 泛化 |
|---|---|---|
| S₀ = 0（裸奔） | 0% | 0% |
| **RAG-oracle（事实进 prompt）** | **98%** | — |
| S₀ @ 100 事实（33 epoch） | 12% | 2% |
| S₀ @ 300 事实（11 epoch） | 13% | 8% |
| S₀ @ 500 事实（9 epoch） | 8% | 0% |
| S₀ @ 100，逐字问题（无换问法） | 5% | — |
| S₀ @ 100，边界修复后逐字 | **12%** | — |

能力保持探针：mem-500 加载后"法国首都"→Paris ✓、翻译 ✓，
但自我介绍进入 "My name is" 吸引子——**通用能力部分受损**。

## 3. 三个结论

1. **S₀ 注入事实知识的上限极低（≤13%）且不随训练量/知识量增长。**
   容量曲线平坦 + mem-100 训练 loss 归零但换问法召回仅 12% +
   逐字召回也只有 5%——注入的是脆弱的表层绑定，不是可复用的知识。
   而 RAG-oracle 98% 证明底座**使用**知识的能力完好无损。
2. **知识在上下文和权重里，不在初始状态里。** 这与"风格/任务模式可以
   烘焙进 S₀"（neko 100%、zh2en 100%）形成鲜明对照——S₀ 适合编码
   *行为模式*，不适合编码*长尾事实*。
3. **过程中发现并修复了一个更重要的 bug**（见 §4）：修复后逐字召回
   5%→12%、泛化 2%→12%，并使 S₀ 线性组合从"相变"变为"可行"
   （见 [state-arithmetic2.md](state-arithmetic2.md)）。

## 4. 附带发现：字节级分词边界错位 bug

原 `build_example` 对 "prompt+答案" 整句编码。字节级词表的贪婪匹配会产生
**跨 prompt/答案边界的合并 token**（吞掉答案首字符的部分字节），而 `prompt_len`
掩码按单独编码的 prompt 长度计算——**掩码边界与 token 边界错位**。
后果链：训练 loss 可以归零（分布内拟合）→ 但注入的是"残缺字节续写"模式 →
推理召回崩溃 → 且这种 S₀ 在线性混合时产生相变（第一轮算术结论的伪影）。

修复：prompt 与 completion **分段编码后拼接**；serving prompt 与训练侧
严格对齐（含尾随空格）。修复效果：逐字召回 5%→12%、泛化 2%→12%、
人格质量提升、S₀ 可组合性恢复。

**通用警告**：任何"prompt+completion 掩码训练 + 字节级分词器"的组合
（state tuning、前缀调优、乃至普通 SFT）都可能踩中这个坑。诊断方法：
分别评测"逐字问题"与"换问法问题"——两者同时远低于训练 loss 所暗示的
水平，即为边界错位信号。

## 5. 对 stateswap 的定位影响

- **S₀ 的正确用途**：风格 / 任务模式 / 行为先验（已验证 100% 可烘焙、
  可热切换、修复后可组合）。
- **知识的正确用途**：RAG / 上下文 / 权重训练。stateswap 的
  会话状态缓存 + 人格热切换 + RAG 是互补而非竞争关系。
- StateBench（`src/stateswap/benchsuite.py`）已将上述评测协议产品化：
  风格命中率 / 知识注入 / 能力保持三类探针 + scorecard 输出。

## 复现

```bash
python scripts/gen_knowledge_corpus.py
python -m stateswap.train --model models/rwkv7-1.5b-world-hf \
    --data data/knowledge_qa_100.json --out personas/mem-100 --steps 3000
python scripts/eval_memory.py
python scripts/eval_memory_control.py   # 逐字对照
```
