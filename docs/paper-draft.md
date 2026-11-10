# The Geometry of RWKV Initial States: Persona Tuning, Phase Transitions, and Low-Rank Surgery

**Technical report — draft v0.1** · stateswap @ commit `HEAD` · all numbers reproducible from `docs/*.json` via the scripts in `scripts/`

---

## Abstract

We treat the RWKV-7 initial recurrent state S₀ — one 64×64 matrix per head per layer, ≈12.6 MB at the 1.5B scale — as the *sole trainable parameter* of a fully frozen base model, and study through six controlled experiments what it can and cannot do. Style and task modes bake perfectly (catgirl-style hit rate 12%→100%; unprompted zh→en translation 100%), while factual knowledge does not (≤13% recall regardless of corpus size; RAG-oracle 98%): **S₀ is a behavior prior, not a knowledge store**. Layer-group ablation locates task modes in the front third of layers and style in a distributed, late-layer-weighted subspace; per-head SVD shows a persona needs only ≈3–4 dimensions per head behaviorally, though 90% of its Frobenius energy needs 14–20. Mixing two trained states along the persona axis yields a **two-transition phase diagram** — pure-style for α∈[0.65,1], pure-task for α∈[0,0.5], and a narrow coexistence window at α≈[0.55,0.65] — with per-prompt attractor selection at the boundaries and a striking geometry–behavior decoupling: endpoint cosine similarity moves near-linearly while behavior jumps twice. Composition by training (warm start from a style donor, then task data) produces the opposite failure: task 100%, donor style 0%, even when the back 16 layers are bit-frozen — behavioral mode is selected by the front layers regardless of preserved style content. Low-rank directional edits (subspace removal, targeted injection, 18 configurations) confirm the picture: edits navigate the phase diagram but cannot surgically decompose it — removal from either direction is a factory reset. We release **stateswap**, a serving stack built on these findings: O(1) session state (12.78 MB constant), ≈3 ms persona hot-swap, 6.29× batched multi-session decode, int8 (4×) and rank-4 (7.9×) persona compression, a card→corpus→gated-persona factory, and a zero-dependency WebUI with a live 24-layer state monitor.

---

## 1. Introduction

Injecting a behavior into a frozen large language model has three canonical injection sites, each with a different cost model: **weights** (LoRA — GBs per persona, training required), **context** (system prompts — zero training, per-request token cost), and — specific to recurrent architectures — the **initial state**. RWKV-7 maintains a per-layer, per-head matrix recurrence state S that evolves with every token; standard inference cold-starts it at zero. Nothing about the architecture requires zero: S₀ can be treated as an explicit, trainable, *swappable* tensor. This affordance has no Transformer analogue — a Transformer's "initial KV cache" is not an object one can hold, compress, or hand to another process.

We systematize this affordance. Our contributions:

1. **The first systematic capability map of S₀** — what bakes (style, task modes), what cannot (facts), and what deep injection costs in base-model capability (a three-way S₀/LoRA/prompt comparison).
2. **An anatomy of a persona**: layer-group ablation (task = front third, style = late-weighted distributed) and per-head SVD (persona ≈ 3–4 dims/head behaviorally).
3. **A phase diagram of the persona axis** between two trained states: 13-point sweep, coexistence window, per-prompt attractor selection, geometry–behavior decoupling.
4. **Interventional validation via low-rank surgery**: 18 edit configurations showing that edits navigate but cannot decompose the phase diagram.
5. **stateswap**, an open serving stack that turns these findings into product mechanics: O(1) sessions, ~3 ms hot-swap, batched decode, persona compression, a gated persona factory — and **MemoPet**, a spin-off whose pet "brain" is literally a persisted S₀ (save/fork/dream).
6. Two reproducible engine bugs with diagnostics (fla's fused-recurrent S₀ gradient drop; a byte-level tokenizer mask-boundary bug that silently destroys composability).

## 2. Background and Related Work

**RWKV-7 recurrence.** RWKV-7 "Goose" is a linear-complexity RNN family with matrix-valued states: each layer maintains H heads of 64×64 state matrices updated by a generalized delta rule with vector-valued gating and in-context learning rates. Training is parallel (chunked recurrence); inference is O(1) per token with constant memory. RWKV-7's state tracking recognizes all regular languages, exceeding what Transformers can do under standard complexity conjectures (TC⁰).

**State tuning lineage.** Preen demonstrated state tuning on Mac/MLX; RWKV-PEFT ships it officially; State Tuning (arXiv:2504.05097) uses state optimization for test-time scaling with an observer-distillation pipeline. Our setting is simpler: offline training of S₀ on small corpora, then serving.

**Merging and state mixing.** Task-vector arithmetic (model soups, mode connectivity) operates in weight space. State Soup (2406.08423) and Soupability (2505.24033) move mixing to the state level on Mamba, reporting *smooth* interpolation benefits for ICL skills. Our Study 5 finds the opposite qualitative behavior for RWKV-7 task/persona attractors — a two-transition phase diagram — and Study 3 traces an earlier "sharp transition" observation to a training-pipeline bug, not to the geometry.

**Model editing.** ROME/MEMIT edit MLP weights in Transformers; ROME-on-Mamba (2404.03646) locates and rank-one-edits factual associations in Mamba; DREAMSTATE (2601.19221) generates and edits RWKV states with conditional diffusion, explicitly flagging "RNN states as editable knowledge" as an open problem. Our Study 6 occupies the precise gap between them: *directed, training-free, rank-k subspace operators on a trained S₀*.

## 3. Experimental Setup

**Base model.** RWKV7-World-1.5B (bf16, fully frozen, 24 layers × 32 heads × 64×64 state; S₀ = 3,145,728 parameters ≈ 12.58 MB fp32). All experiments post-date a tokenization-boundary bug fix (§6.2) unless stated — the bug's effect on composability is itself a finding.

**Personas.** `neko-1.5b` (catgirl style; 3,095 QA pairs; style hit 100%) and `zh2en-1.5b` (unprompted zh→en translation; 7,369 WMT pairs; 100% English rate). Training: AdamW, peak lr 1e-4 cosine to 1e-5, cross-entropy on assistant spans only, fresh S₀ cold-start per sample.

**Protocol.** Greedy decoding (temperature 0) unless stated; fresh session per probe; bounded context replay disabled for probes (context_replay=0); style hit rate = marker presence (喵/主人/（/小鱼干…); translation rate = CJK-free English output; raw replies and per-layer statistics preserved as JSON artifacts (`docs/*.json`) alongside one-command probe scripts (`scripts/probe_*.py`). A degeneration guard (turn-start state snapshot + character 4-gram uniqueness check + rollback) prevents polluted generations from entering any measured state.

## 4. Study 1 — What S₀ Can Bake, and What It Cannot

**Style and task modes bake perfectly.** Style hit rate 12–25% (cold) → 88–100%; at 1.5B, 12%→100% with 4,000 steps. Unprompted translation reaches 100% on held-out bare Chinese sentences. Data scale is a hard requirement: 50 pairs of bare-format translation data reach zero training loss yet only *memorize* (any paraphrase fails); 7,369 pairs generalize to 100% — task invariance needs both the training distribution and the data mass.

**Facts do not bake.** A controlled synthetic-knowledge study (100/300/500 facts, up to 33 epochs, training loss ≈ 0) yields ≤13% recall, *decreasing* with corpus size, versus 98% for a RAG oracle over the same facts. General capability is partially retained (capital-of-France survives) but self-description collapses into attractor text. Conclusion: **S₀ is a behavior prior, not a knowledge store**.

**Three-way comparison (S₀ / LoRA / system prompt).** All three inject style (100%/100%/80% at LoRA's 4.5× parameter budget). But depth injection overrides base capability: "capital of France" is answered in catgirl cadence under S₀ and LoRA, while the system prompt preserves it verbatim. Serving economics differ by an order of magnitude per persona: S₀ hot-swap 13.7 ms / 12.6 MB; LoRA one 3 GB model directory per persona; system prompt 1,731 tokens per request.

## 5. Study 2 — Anatomy of a Persona

**Layer addresses.** Zeroing front-third layers (L00–07) kills translation outright (0%) but keeps 80% style; zeroing back-third (L16–23) costs 60% style; the middle 8 layers are dispensable for both. Task mode lives in the front third; style is late-layer-weighted and distributed — no single third suffices for it alone.

**Per-head dimensionality.** 90% of Frobenius energy needs rank 14–20, but *behavior* survives rank-3 truncation at 100% (style and translation): a persona needs only ≈3–4 dimensions per head. Most state mass is behaviorally inert, which also explains the attractor-like transitions of Study 5 — a persona is a handful of directions in state space, not a volume. Rank-4 factored personas ship as 12.58 MB → 1.59 MB artifacts, behaviorally lossless.

## 6. Study 3 — Composition by Arithmetic (Two Rounds)

**Round 1 (0.4B, buggy pipeline).** Mixing S = α·S_neko + (1−α)·S_zh2en at α ∈ {1, 0.75, 0.5, 0.25, 0} showed a sharp transition between 0.75 and 0.5: pure style → pure translator, plus "un-addable" additive sums.

**The boundary bug.** Round 1 personas were trained with a byte-level tokenizer mask misalignment: prompt+completion encoded as one string lets greedy BPE merge across the boundary while the mask boundary is computed from the separately-encoded prompt. Training loss reaches zero, but the model learns *fragment continuation* — verbatim recall 5%, paraphrase 0% — and mixed S₀s encode conflicting surface fragments, which collapse to a single mode. After the fix (segmented encoding; boundary tokens belong to the completion), recall improves 2.4× and — dramatically — **composability is restored**.

**Round 2 (1.5B, fixed).** The global 50/50 mix keeps both capabilities (style 100% + translation 100%); front-12/back-12 layer mixes show the front half dominating expression. This reversal is retained as a methodological warning: **any masked-prefix training on byte-level tokenizers must be boundary-audited**, and experiment conclusions must be re-run after pipeline fixes.

## 7. Study 4 — Composition by Training (Inheritance)

Warm start S₀ ← style donor, then train on task data:

- **Naive (all layers trainable), 2,000 steps:** task 100%, donor style 0% — every input, including chat, is translated. The task attractor captures all behavior.
- **Layer-masked (--train-layers 0:8; back 16 layers frozen bit-exact at donor values):** task 100%, style **0%**. The donor's style content survives perfectly in the state — per-layer cosine to the donor is 1.000 in L08–23 — yet is never expressed.

This upgrades the anatomy report's correlational claim ("front layers dominate expression") to a **causal manipulation**: preserving a behavior's layer address does not preserve the behavior, because *which mode is expressed is selected by the front layers*. Training-based composition is an incremental-learning mechanism (it writes new capability into specified layers), not a composition mechanism — arithmetic mixing (Study 3/5) remains the way to keep both.

## 8. Study 5 — The Phase Diagram

A 13-point α sweep over the persona axis (fixed pipeline, 1.5B):

| phase | α range | style | task |
|---|---|---|---|
| pure style | [0.65, 1] | 100% | 0% |
| **coexistence** | [0.55, 0.65] | 87.5% (α=0.6) | 75% |
| pure task | [0, 0.5] | 0% | 100% |

The center α≈0.58 sits style-side of the window. **Per-prompt selection (E6)**: no reply is ever blended — each prompt flips the whole generation to one phase, with prompt-dependent thresholds near the boundaries, and base-model "other" leakage exactly at the edges. **Geometry–behavior decoupling**: endpoint cosine similarities move near-linearly in α while the behavior order parameter jumps twice — the state is a basin selector, and basins have walls. **Session locking**: the *first* exchange decides the phase at α=0.5 (12.5% fresh vs 100% in-session) — in-context evidence outcompetes the initial-state prior. Equilibrium hysteresis protocols failed for measurement reasons (blend-walk degeneration); fresh-session + depth-controlled history is the proposed protocol. Temperature robustness holds (T=0 vs 0.7 thresholds align).

**Against State Soup**: smooth interpolation benefits on Mamba ICL skills versus a two-transition diagram on RWKV-7 task/persona attractors. Ranked hypotheses: (a) ICL skill vectors vs. strong task attractors; (b) RWKV-7 delta-rule geometry vs. Mamba's; (c) endpoint orthogonality (cos 0.02–0.14) of our persona pair.

## 9. Study 6 — Low-Rank Surgery

Operators on a trained S₀: `remove_subspace(s, ref, k)` = (I − UUᵀ) s (I − VVᵀ) with U/V from ref's per-head SVD (surgical excision of the components living in ref's top-k subspace); `inject(s, direction, λ, k)` = s + λ·rank-k(direction); `rank_k` truncation. 18 configurations against the coexistence mixture:

- **Removal in either direction, any k → both capabilities collapse to the base model** (factory reset). The coexistence mixture is a measure-zero knife edge: touching either constituent's subspace destroys the balance.
- **Targeted injection (cos 0.99 to the target S₀) → full flip into that attractor** at both capability readouts.
- Scalar subtraction works when the goal is one-directional (S_neko − S_zh2en keeps style, kills translation).

**Pitfall**: a singular-value broadcasting bug (per-head SVD applied to the batched matrix) was caught by a reconstruction-equivalence test — geometric checks alone are insufficient; every edit must pass behavioral probes (§3 protocol), because geometry–behavior decoupling (Study 5) means small edits can flip modes.

**Takeaway**: persona assets can be *navigated* (mixing, injection, warm start) but not *decomposed* (removal). Combined with Study 4, the asymmetric conclusion is: write capabilities in, cannot edit behaviors out.

## 10. System — stateswap

A serving stack that operationalizes the findings:

- **Engine**: per-session persistent state (12.78 MB constant, vs. O(T) KV caches); ~3 ms persona hot-swap; batched multi-session decode (6.29× at B=8); degeneration guard (turn-start snapshot + 4-gram uniqueness + rollback, so poisoned generations never enter long-term state); bounded context replay (bridging the 1.5B fact-retention ceiling without breaking O(1)); **attractor seeding** (a hidden in-context persona exchange that locks a fresh session into its attractor — first-exchange phase selection, Study 5, turned into a product feature).
- **Compression**: int8 (4×) and rank-4 (7.9×) persona artifacts, behaviorally lossless.
- **Persona factory**: card → LLM-generated corpus → multi-turn S₀ training → evaluation gate (style hit rate + capability-retention probes + degeneration check) → live registration. A failed gate refuses registration.
- **Engineering notes** (full details in the repository): fla/Triton pitfalls on Windows + Blackwell (sm_120); the fused-recurrent path dropping S₀ gradients in eval mode (train-mode chunk path required); the mask-boundary bug of §6.2; a CI dependency-drift postmortem; SSE-abort lock semantics.

**MemoPet** (spin-off): a virtual pet whose brain is a persisted S₀ — save/fork/dream/time-travel mechanics built directly on state persistence, demonstrating the "persona as an MB-scale asset" economics outside the chat frame.

## 11. Discussion

The six studies compose into one picture: **the initial state is a basin selector, not a content carrier.** Modes (style, task) bake and combine because they are attractor directions; facts do not bake because a 12.6 MB state cannot hold a knowledge base; removal fails because deleting a direction collapses the basin balance rather than excising a behavior; the front layers select which basin is expressed regardless of what content survives elsewhere.

Implications: (i) recurrent-model behavior editing should target *basin selection* (injection, gating, seeding), not content excision; (ii) persona-as-asset economics are real — MB-scale, hot-swappable, compressible, forkable; (iii) RWKV-8's "mixed/smaller state" roadmap should account for phase structure — smaller states must preserve the basin geometry that makes personas possible.

## 12. Limitations

Single persona pair (neko × zh2en) on a single base (1.5B); single-seed runs; heuristic metrics (marker/substring matching) rather than human evaluation; no Mamba control for the phase diagram; equilibrium hysteresis unmeasured (protocol in §8); editing studied only against the coexistence mixture.

## 13. Related Work (full)

State Soup [2406.08423]; Soupability [2505.24033]; ROME-on-Mamba [2404.03646]; DREAMSTATE [2601.19221]; State Tuning on RWKV-7 [2504.05097]; TTT layers; Dynamic Evaluation (Krause 2018); When Scaling Meets Finetuning [ICLR 2024]; model soups and task vectors; mode connectivity; RWKV-7 "Goose" [2503.14456]; Mamba-2; HART [2610.12883] (placeholder idling decay in RWKV states); SANE [2608.22354] (long-range state stability); WriteSAE [2605.12770] (rank-1 SAE atoms on RWKV-7 cache writes).

## 14. Reproducibility

Repository `stateswap` (GitHub, MIT): one-command probe per study (`scripts/probe_*.py`), raw JSON artifacts per study (`docs/*.json`), trained personas committed (12.6 MB each, with int8/rank-4 variants), CI running the CPU test suite. Hardware for all results: one RTX 5070 Ti Laptop (12 GB), Windows 10, torch 2.11 cu128, fla 0.5.2. The companion repository `memopet` demonstrates the state-as-asset mechanics (persistence, forking, dreams) outside the chat frame.

---

*Draft status: v0.1 — all section content present, prose passes pending a technical edit; figures to be generated programmatically from `docs/*.json` per the figure table in `docs/paper-outline.md`.*
