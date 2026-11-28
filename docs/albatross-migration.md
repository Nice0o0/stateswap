# Albatross 迁移设计

> 状态：**设计稿，未实施**。前置条件（CUDA Toolkit + MSVC）尚未满足，
> 本文件的目的就是在装工具链之前把改造范围、风险点和验收标准定清楚，
> 这样环境就绪后可以直接动手，不会边想边改。
>
> 起因：fla 0.5.2 的 RWKV-7 被上游明确标注为「未与参考实现对齐、性能明显更差」，
> 且其 decode 路径（`fused_recurrent`）实测输出分布劣化——同一 prompt 同一状态下
> prefill 与 decode 两条路径 `max|Δlogit| = 0.25`、`mean = 0.047`，decode 的 top-8
> 里会出现 U+FFFD（`scripts/probe_chunk_vs_fused.py`）。Albatross 是官方推理实现，
> 同时解决「数值正确性」与「dispatch 受限的解码速度」两个问题。
>
> **本文件基于对 Albatross `faster3a_2607/rwkv7_fast_v3a.py` 源码的阅读**，
> 但该文件在抓取时被截断，几处标记为「待确认」的点必须在环境就绪后用真实源码复核。

---

## 1. 两边状态结构的对应关系

这是整个迁移的技术核心：**Albatross 的 state 是 `[shift, WKV, elapsed]` 三元组，
以显式参数传入传出**，不持有 fla 那种带 dict 的 Cache 对象。

| fla（现状） | Albatross | 形状对应 |
|---|---|---|
| `cache[i]["recurrent_state"]` | `state[1]`（WKV） | fla: `(B,H,64,64)` 逐层<br>Albatross: `(L,B,H,64,64)` 整体<br>**同一个对象，只是堆叠方式不同** |
| `cache[i]["conv_state"]` | `state[0]`（shift）第 0 片 | fla: `(B,C)` 逐层<br>Albatross: `(L,2,B,C)`（2 = token-shift 的前两个 token） |
| `cache[i]["ffn_state"]` | `state[0]`（shift）第 1 片 | 同上 |
| `offset`（计数器） | `state[2]`（elapsed） | Albatross 是 `(B,)` int32，fla 是每层 `offset` |

**结论：S₀ 的形状 `(L,H,64,64)` 与 Albatross 的 WKV 状态 `(L,B,H,64,64)` 在
`B=1` 时完全一致。** 也就是说「人格 = 张量」这个核心抽象不需要任何转换——
`persona.s0.unsqueeze(1)` 就是可以直接塞进 `state[1]` 的对象。这是迁移可行性的
最重要依据，也让 `Persona`／`Session`／护栏／SSE 这些纯张量逻辑完全不用改。

**内存口径实测吻合**（1.5B，B=1）：

| 实现 | 组成 | 合计 |
|---|---|---|
| fla | 24 层 × (recurrent fp32 + conv bf16 + ffn bf16) | **12.78 MB** |
| Albatross | shift `(24,2,1,2048)` fp16 = 0.20 MB + WKV `(24,1,32,64,64)` fp32 = 12.58 MB | **12.78 MB** |

两者**完全相等**，不是巧合：`shift` 的 `2` 个 channel 正是 fla 的 `conv_state`
与 `ffn_state`，`WKV` 正是 `recurrent_state`。这也意味着 README 里「每会话
12.78 MB」这个数字在迁移后**不需要改**。

---

## 2. 必须先解决的硬约束：DeltaLog

这是**整个迁移最大的风险点**，且原文说得很明确：

> a caller "must execute a complete static 0..M-1 cycle before treating state as
> materialized or exposing it to clone/reorder/save logic"

DeltaLog 是 WKV 的可选模式：它在 `0..M-1` 的相位里累积工作区，只在周期结束时
把日志合并进真正的状态。`WKV_DELTALOG_M` 默认 **0 = 未启用**。

| 场景 | 后果 |
|---|---|
| `merge_interval=0`（默认） | 走普通逐 token 路径，**状态每步都是材料化的** → 我们的用法没问题 |
| 启用 DeltaLog（M∈{2,3,4,6,8}） | 周期中途的状态**不是材料化状态**，且**没有 session 记录的外部张量会被当成 phase 0 重新开始** |

**对本项目的直接影响**：我们的整个模型建立在「状态随时可存、可切、可 fork」
之上（这是 stateswap 与 MemoPet 的产品论点）。如果必须启用 DeltaLog（比如为了
速度），那么存档/fork/快照回滚都会碰到「非材料化状态」的语义问题。

**决定：第一版迁移强制 `merge_interval=0`（不启用 DeltaLog）**，把
「DeltaLog 下的状态持久化」列为独立的后续研究项。这一条必须写进代码注释，
否则未来有人为了性能打开它，会静默破坏存档语义。

---

## 3. 需要改动的代码（按文件）

### 3.1 `src/stateswap/s0.py` — 状态容器层

| 函数 | 现状 | 迁移后 |
|---|---|---|
| `make_cache(model, s0, ...)` | 返回 fla `Cache`，逐层 `update` | **新增** `make_state(model, s0, batch_size)`：返回 `[shift, WKV, elapsed]`，WKV = `s0.unsqueeze(1)` |
| `S0.load_stacked` / `stacked` | 与 fla 无关 | **不改**（纯张量） |
| `load_s0_payload` | 与 fla 无关 | **不改** |

`make_cache` 保留（训练侧继续用 fla，见 §5），新增的 `make_state` 供推理用。

### 3.2 `src/stateswap/engine.py` — 会话状态与解码

| 位置 | 现状 | 迁移后 |
|---|---|---|
| `Session.cache: Cache` | 类型注解 | 改成 `state: list[Tensor]` |
| `_cache_for()` | `make_cache` | `make_state` |
| `Session.memory_mb()` | 遍历 `cache[i]` 的三个键 | 遍历 `state[1].numel()` 等（数值应等价） |
| `swap_persona()` 的 `keep_context=True` | 逐层 `cache.update(...)`，清零 conv/ffn | 重建 state，shift 置零，**保留 elapsed**（对应原来的 offset=0？待确认） |
| `_stream_prompt()` 的 prefill | `model(input_ids, past_key_values=cache, use_cache=True)` | `model.forward_from_x(...)` 或 `forward(...)`（**签名待确认**） |
| `_stream_prompt()` 的逐 token | 同上，`([[nxt]])` | `model.forward_deltalog_step(tokens, state)`（`merge_interval=0`） |
| 护栏快照 | 逐层 `.to("cpu")` 三个键 | `[t.to("cpu") for t in state]`（**三个张量，比原来简单**） |
| 护栏回滚 | 逐层 `cache.update(...)` | `state` 三个张量 `copy_` 回去 |
| `batch_chat()` | 沿 batch 维 `torch.cat` / `[b:b+1]` | Albatross 的 WKV 本身就是 `(L,B,H,64,64)`，**批量支持是原生的**，可能比现在更简单 |

**预计难点**：`swap_persona` 的 `offset` 语义。fla 的 `offset` 是每层的 token 计数
（用于 token-shift 记账），Albatross 用全局 `elapsed`。两者的等价性必须在
环境就绪后逐项验证——这是 §4 回归清单里的第一项。

### 3.3 `src/stateswap/server.py`

几乎不用改：它只通过 `engine` 的公开方法访问状态，而 `session_memory_mb`
这类数值是 `Session.memory_mb()` 算出来的。

唯一的改动点：`/v1/sessions/{id}/state-stats` 走 `_recurrent_4d(cache[i][...])`，
迁移后直接读 `state[1][i]`，**代码更短**。

### 3.4 不需要改的（重要）

以下全部是纯张量逻辑，与引擎无关，**一行都不用动**：

- `Persona` / `Session` 的字段（除 `cache` 类型）
- `arithmetic.py`（插值、加法、减法）
- `editing.py`（`remove_subspace` 的 `layers`/`beta`）
- `quant.py` / `lowrank.py`
- `safety.py`
- 退化护栏的**判定逻辑**（只改快照/回滚的实现手法）
- `factory.py` 的造数与评测（它通过 Engine 的公开接口调用）
- 整个 WebUI

---

## 4. 验收标准（迁移算成功的最低条件）

### 4.1 数值一致性（最高优先级）

| 检查 | 判据 |
|---|---|
| **跨引擎 logits 对齐** | 同一 S₀ + 同一 prompt，Albatross 与 fla 的 logits `max\|Δ\|` 必须显著小于 fla 内部两路径的差距（0.25）。这本身就是「fla 未对齐」的独立证据 |
| 逐 token 解码 | 贪心 24 步产出**不同的 token**（回归 §6.3 的 graph 陷阱：状态必须真的递推） |
| `swap_persona` | 切换后人格行为正确，`keep_context=True` 时历史仍可回放 |

### 4.2 行为回归（用现有探针，不新建）

| 探针 | 期望 |
|---|---|
`scripts/probe_editing.py --smoke` | 与迁移前一致（基线 + 一次切除） |
| 8+8 风格/翻译探针 | 风格命中、英文输出率与 fla 版**同一量级**（允许因引擎改进而提高，不允许崩塌） |
| `probe_longconv.py` | 退化轮数与迁移前可比 |
| 存档/fork 往返 | 存盘→重载→继续对话，行为与不断点一致 |

### 4.3 性能（次要，但要报）

| 指标 | fla 现状 | 期望 |
|---|---|---|
| decode | 47 ms/token（GPU 仅 0.1 ms） | 官方称 5090 上 7.2B 达 145 tok/s；0.4B/1.5B 应显著优于现状 |
| prefill | 稳态 ~60 ms | 不应变差 |

---

## 5. 两种迁移策略

### 策略 A：推理换 Albatross，训练继续用 fla（推荐）

**理由**：训练侧的 fla 梯度 bug 已经用 `model.train()` 绕过（§4.1），且训练
不产生对外报的绝对数字；而推理侧才是数值正确性与速度的关键。

**代价**：同时加载两套权重格式（HF safetensors + 官方 .pth），显存翻倍。
1.5B 的 bf16 权重约 3 GB，12 GB 卡上还剩 9 GB，**可行但不宽裕**。
缓解：迁移完成后可只留 .pth，把 HF 转换产物降级为「训练专用」。

### 策略 B：训练也换（Albatross 不做训练，需换回参考实现）

Albatross 是纯推理库（`torch.load(MODEL_PATH)`，无训练循环）。训练继续留在 fla
才是现实选择，所以 **B 不可行**——这不是取舍，是事实。

结论：**只能走 A**。

---

## 6. 待确认清单（环境就绪后逐条核对）

这些是我读被截断的源码时**无法确认**的点，必须在装好工具链后打开真实文件验证：

1. `forward` / `forward_from_x` 的**确切签名**是否存在普通（非 DeltaLog）路径；
2. `elapsed` 与 fla `offset` 的语义是否等价（token-shift 记账）；
3. `state[0]`（shift）两个 channel 的**确切含义**——是否就是 fla 的
   `conv_state` + `ffn_state`，还是 token-shift 的「前两个 token 的 x」；
4. 是否有官方提供的 state save/load 辅助函数（影响存档代码要写多少）；
5. `load_weights` 期望的 `.pth` key 命名是否与我们 `downloads/` 里的
   `RWKV-x070-World-*.pth` 直接匹配（应当是，因为那就是 BlinkDL 的格式）；
6. `batch_chat` 的批量语义（`(L,B,H,64,64)` 是原生支持，但要确认 per-sample
   完成状态如何管理）。

---

## 7. 实施顺序（每步都可独立回退）

1. **构建验证**（半天）：装 CUDA Toolkit 12.8 + VS Build Tools，单独编译
   `rwkv7_wkv_fp16_v2` 一个 op。**失败则整个计划终止**，成本可控。
2. **状态容器**（半天）：`make_state()` + 单元测试（形状、与 S₀ 的对应、
   `memory_mb` 数值等价）。此时不碰 engine，旧路径仍可用。
3. **解码路径**（1 天）：`_stream_prompt` 换 Albatross，跑 §4.1 的数值一致性。
4. **会话机制**（半天）：`swap_persona` / 护栏快照回滚 / `state-stats`。
5. **行为回归**（半天）：§4.2 全部探针。
6. **批量解码**（半天）：`batch_chat` 换原生批量。
7. **文档**：benchmarks 补新数字、README 改「Engine」段，并在
   `docs/benchmarks.md` 记录 fla → Albatross 的差异实测。

**总计约 3–4 天**（不含工具链安装）。每步都有独立的验收点，任何一步失败
都可以停在「旧路径仍工作」的状态。

---

## 8. 如果构建验证失败，替代方案

1. **只换 decode kernel**：用 Albatross 的 `cuda/` 源文件单独编译
   `rwkv7_wkv_*`，塞回 fla 的 `fused_recurrent` 位置——风险是接口不兼容。
2. **等 fla 对齐**：跟踪上游 issue，这是零成本但不可控。
3. **放弃迁移，只记录 caveat**：现状已经在 README 顶部写清楚了
   「相对结论可信、绝对数字偏低」，不迁移也能继续做研究，只是少了一条性能故事。
4. **借 Albatross 做交叉验证**：哪怕不迁移，只要能编译一个 op 并在小样例上
   跑通，就能给出「fla vs 官方」的**逐 logit 对照**——那本身就是一条扎实的独立结论，
   而且成本远低于完整迁移。

第 4 条值得强调：**它把「迁移」这个 3–4 天的工程任务，降级为「半天的实验」**，
而产出的科学价值（验证 fla 偏差的具体幅度）几乎是一样的。
