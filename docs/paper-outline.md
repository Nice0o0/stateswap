# Paper Outline — The Geometry of RWKV Initial States

**Working title**: *The Geometry of RWKV Initial States: Persona Tuning, Phase Transitions, and Low-Rank Surgery*

**One-line thesis**: The RWKV-7 initial recurrent state S₀ is a trainable, servable, and partially editable behavior asset — but its behavior is selected by attractor dynamics, not linear geometry: it can be navigated, not smoothly blended or surgically decomposed.

---

## 1. Abstract

We treat the RWKV-7 initial state S₀ (one 64×64 matrix per head per layer, ~12.6 MB at 1.5B) as the sole trainable parameter of a frozen base model and study, through six controlled experiments, what it can and cannot do. Style and task modes bake perfectly (12%→100% style, 100% unprompted translation); factual knowledge does not (≤13%). The persona axis between two trained states is a two-transition phase diagram with a narrow coexistence window, per-prompt attractor selection, and a geometry-behavior decoupling. Low-rank directional edits navigate the phase diagram but cannot surgically decompose it: removal is a factory reset. We release stateswap, a serving stack that exploits these findings (O(1) session state, ~3ms persona hot-swap, 6.29× batched decode, a card-to-corpus-to-gated-persona factory).

## 2. Introduction

- Frozen-base behavior injection: LoRA (weights), system prompt (context), S₀ (initial state) — three injection sites, three cost models.
- RWKV's initial state as an explicit, swappable tensor — a unique affordance vs Transformers.
- Contributions: (a) first systematic map of S₀ capabilities/limits; (b) phase diagram of the persona axis; (c) interventional validation via low-rank editing; (d) serving system + factory pipeline; (e) two engine bugs with reproducible diagnostics.

## 3. Background

- RWKV-7 recurrence and state semantics (delta rule, vector gating, in-context learning rates; constant-memory inference).
- State tuning lineage: Preen (MLX/Mac), RWKV-PEFT, State Tuning (test-time scaling on RWKV-7, 2504.05097).
- Model merging/task vectors (soups); State Soup & Soupability (state-level mixing on Mamba).
- Model editing: ROME/MEMIT; ROME-on-Mamba (2404.03646); DREAMSTATE (2601.19221).

## 4. Setup

- Base: RWKV7-World-1.5B, bf16, fully frozen; personas: neko-1.5b (3,095 pairs), zh2en-1.5b (7,369 WMT pairs), all trained post-tokenization-boundary-fix.
- Metrics and protocol: style hit rate (markers), English rate (bare-zh sentences), greedy decoding, fresh sessions, context_replay=0; degeneration guard; per-probe raw replies preserved (docs/*.json).

## 5. Study 1 — What S₀ Can Bake, What It Cannot

- Style: 12–25% → 88–100% (1.5B: 12%→100%); translation: 100% unprompted.
- Compressed memory: facts ≤13% regardless of scale; RAG-oracle 98% → S₀ is a behavior prior, not a knowledge store (docs/compressed-memory.md).
- Three-way comparison: S₀/LoRA/prompt all inject style (100/100/80%); deep injection overrides general capability, prompt preserves it; serving cost 13.7ms vs 3GB per persona vs 1,731-token prefix.
- Data point for the discussion: scaling curve (§Study 7 appendix, docs/scaling.md).

## 6. Study 2 — Anatomy of a Persona

- Layer addresses: task mode in front third (zeroing L00–07 kills translation), style late-layer-weighted and distributed (zeroing L16–23 costs 60%), middle 8 dispensable.
- Per-head dimensionality: ~3–4 dims behaviorally; 90% Frobenius energy needs 14–20 → most state mass is behaviorally inert; rank-4 factored personas 12.58→1.59MB lossless.

## 7. Study 3 — Composition by Arithmetic

- Round 1 (0.4B, buggy pipeline): sharp transition between α=0.75 and 0.5.
- The byte-level tokenizer boundary bug (mask misalignment): loss zero, recall 5%; fix restores 2.4× recall and composability — methodological warning for any masked-prefix training with byte-level tokenizers.
- Round 2 (1.5B, fixed): 50/50 keeps both (as measured then) → motivated the
  fine-grained mapping in Study 5, which relocates the robust coexistence
  window to α ≈ [0.55, 0.65] and shows exactly 0.5 has already collapsed.

## 8. Study 4 — Composition by Training (Inheritance)

- Warm start from style donor + task data: task 100%, donor style 0% — every input translated (task attractor captures behavior).
- Layer-masked warm start (front 8 only, back 16 bit-preserved): still 0% style — the style content survives perfectly in the state yet is never expressed; behavioral mode is selected by the front layers. Anatomy correlation upgraded to causal manipulation.

## 9. Study 5 — The Phase Diagram

- 13-point α sweep: pure style [0.65,1], pure task [0,0.5], coexistence [0.55,0.65] (α=0.6: 87.5% style + 75% task); center ≈0.58, style-side.
- Per-prompt selection (E6): no blended replies; prompt-dependent thresholds; base-model "other" leakage at boundaries.
- Geometry-behavior decoupling: endpoint cosines near-linear while behavior jumps twice.
- Temperature robustness (T=0 vs 0.7). Session locking: first exchange decides the phase at α=0.5 (12.5% fresh vs 100% in-session). Hysteresis protocols fail (blend-walk degeneration; fresh-session + depth-controlled history proposed).
- State Soup contradiction: smooth on Mamba ICL skills vs two-transition on RWKV-7 task/persona attractors — hypotheses ranked.

## 10. Study 6 — Low-Rank Surgery

- 18 edit configurations on the coexistence mixture: removal (either direction, any k) → both collapse to base model (factory reset); injection (cos 0.99) → full flip into target attractor; scalar subtraction works for the one-direction goal.
- Coexistence as measure-zero knife edge; persona assets can be navigated, not decomposed.
- Operator pitfalls: singular-value broadcasting bug caught by reconstruction-equivalence tests.

## 11. System — stateswap

- Engine: O(1) session state (12.78MB constant), ~3ms hot-swap, batch decode 6.29× (B=8), int8 (4×) and rank-4 (7.9×) compression, degeneration guard, bounded context replay.
- Persona factory: card → LLM corpus → multi-turn training → eval gate (style + capability retention) → live registration; API + zero-dependency WebUI with live state monitor.
- Engineering notes: fla/Triton pitfalls on Windows+Blackwell; mask-boundary bug; CI dependency-drift postmortem.

## 12. Discussion

- Unified picture: attractor dynamics in initial-state space; the state as a basin selector rather than a content carrier (facts fail, modes succeed).
- Implications for recurrent-model behavior editing, serving economics (personas as MB assets), and RWKV-8's "mixed/smaller state" roadmap.

## 13. Limitations

Single persona pair (neko×zh2en); 1.5B scale; heuristic metrics (marker/substring matching); single-seed runs; no Mamba control; equilibrium hysteresis unmeasured.

## 14. Related Work (full)

State Soup 2406.08423; Soupability 2505.24033; ROME-on-Mamba 2404.03646; DREAMSTATE 2601.19221; State Tuning 2504.05097; TTT layers; Dynamic Evaluation (2018); When Scaling Meets Finetuning (ICLR 2024); model soups/task vectors; mode connectivity.

## 15. Reproducibility

Repo stateswap (GitHub): one-command probes per study (scripts/probe_*.py), raw JSON artifacts per study (docs/*.json), personas committed (12.6MB each, int8/rank4 variants), CI green.

---

### 配图清单（从仓库 JSON 现拉）

| 图 | 数据源 | 内容 |
|---|---|---|
| Fig 1 | benchmarks | S₀ vs LoRA vs prompt：风格/能力保持/服务成本 |
| Fig 2 | anatomy.json | 逐层消融 + 逐头谱（任务前层/风格后层/3-4 维） |
| Fig 3 | inheritance.json | 三臂对照 + 冻结层仍 0% 风格 |
| Fig 4 | phase-transition.json | 序参量曲线（相图）+ E6 逐 prompt 翻转表 |
| Fig 5 | phase-transition.json e4 | 几何平滑 vs 行为跳变（脱钩） |
| Fig 6 | state-editing.json | 18 编辑配置的行为落点（导航 vs 格式化） |
| Fig 7 | scaling.json | 风格命中/能力保持 vs 语料量（支撑实验） |

### 篇幅与目标形态

- 首选：arXiv 风格技术报告（8-10 页正文 + 附录协议全文），同步一份中文博客版；
- 数据/图全部从 docs/*.json 程序化生成，与仓库 commit hash 绑定。
