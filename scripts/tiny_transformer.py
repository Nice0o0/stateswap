"""训练一个 2 层玩具 Transformer，并把完整前向过程 dump 成 JSON，供 web/transformer.html 可视化。

为什么不用 GPT-2 / GPT-4：
- 真实模型的注意力是 (96 层 × 128 头 × 4096×4096)，肉眼看不出任何东西；
- 这里的模型是 d=64 / 4 头 / 2 层，每个矩阵小到能逐格读，而**算术完全一样**
  （同一套 Q/K/V、同一套 softmax、同一套残差 + LN + FFN）。

任务：上下文内 induction（in-context induction）
    结构  [BOS] w0..w(n-1) [SEP] (gap) w0..w(k-1)  →  预测 w(k)
    例：猫 追 蝴蝶 穿过 花园 藏进 ｜ 猫 追  →  预测「蝴蝶」

这个任务标准解法需要**两层**协作，也就是 GPT 类模型做 in-context learning 时
用的同一套电路（Anthropic 2022 的 "induction head"）：
    第 1 层：每个位置写下「我后面的词是什么」（shift 头）
    第 2 层：末位查「这个词上次出现在哪」，读出那里的「后继词」（induction 头）
真实模型把同一套电路放大到数万维、上千个头。

**关键设计：三条捷径都必须堵死**，否则模型会用位置查表，induction 电路根本不建。
本脚本踩过两次，都记录在 `docs/transformer-trace.json` 的 ablation 里：
  v1 固定长度          → 400 步 100%，但拆掉第 2 层注意力只掉 13 分（背下了"第 10 位是答案"）
  v2 只随机 n_ctx/n_copy → 答案离 SEP 的距离固定，仍能数位置
  v3（本版）n_ctx / n_copy / gap 三者随机 + 词表 60 → 位置不可预测，只能内容匹配
`--scan` 会把每种配置的「准确率 / 各层消融 / induction 分」打出来，据此选配置。

产出：
    docs/transformer-trace.json  全部中间张量 + 消融 + 逐头诊断
    web/transformer-trace.js     同一份数据包成 window.TRANSFORMER_TRACE（页面可 file:// 直开）

用法：
    python scripts/tiny_transformer.py                  # 训练并导出（CPU 约 1 分钟）
    python scripts/tiny_transformer.py --scan            # 扫描配置，选 induction 最清晰的一组
    python scripts/tiny_transformer.py --check           # 只跑自检，不写文件
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

# 60 个词的词表。为什么这么大：词表小的时候「末位 token → 答案」可以当查表背下来
# （20 词时实测准确率 100% 而 induction 分只有 0.23，注意力一团散），
# 60 词且有 gap 时必须做真正的内容匹配。
WORDS = (
    "猫", "狗", "孩子", "老人", "蝴蝶", "鸟", "小鱼", "花", "灯笼", "风筝",
    "书", "船", "风", "雨", "雪", "月亮", "太阳", "星星", "云", "山谷",
    "追", "看见", "飞过", "穿过", "藏进", "抱着", "数着", "画着", "推着", "背着",
    "森林", "花园", "屋顶", "河流", "码头", "广场", "走廊", "草地", "沙滩", "巷子",
    "红色", "蓝色", "白的", "黑的", "柔软", "坚硬", "冰凉", "温暖", "安静", "吵闹",
    "清晨", "午后", "傍晚", "深夜", "春天", "秋天", "冬天", "夏天", "远方", "这里",
)


@dataclass(frozen=True)
class Task:
    """任务定义 + 由它派生的词表索引。三条随机化是防捷径的核心，见模块 docstring。"""
    words: tuple[str, ...] = WORDS
    min_ctx: int = 4
    max_ctx: int = 6
    max_gap: int = 2

    @property
    def tokens(self) -> tuple[str, ...]:
        return ("<BOS>", *self.words, "<SEP>", "<PAD>")

    @property
    def stoi(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.tokens)}

    @property
    def vocab(self) -> int:
        return len(self.tokens)

    # 最长序列：BOS + max_ctx + SEP + max_gap + (max_ctx-1)
    @property
    def max_t(self) -> int:
        return 2 + self.max_ctx + self.max_gap + (self.max_ctx - 1)

    @property
    def ctrl(self) -> dict[str, int]:
        s = self.stoi
        return {"bos": s["<BOS>"], "sep": s["<SEP>"], "pad": s["<PAD>"]}


@dataclass(frozen=True)
class ModelCfg:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


TASK = Task()
CFG = ModelCfg()


# ---------------------------------------------------------------- 数据

def sample_example(task: Task, rng: random.Random) -> tuple[list[int], int]:
    """返回 (token 序列, 正确下一个词)。三处随机各堵一条捷径：
    n_ctx 变 → 总长不固定；n_copy 变 → 答案离 SEP 不固定；gap 变 → 离末端也不固定。"""
    stoi = task.stoi
    n_ctx = rng.randint(task.min_ctx, task.max_ctx)
    n_copy = rng.randint(2, n_ctx - 1)
    ctx = rng.sample(task.words, n_ctx)                 # 不重复取词，保证"上次出现"唯一
    gap = rng.sample(task.words, rng.randint(0, task.max_gap))
    base = [stoi[w] for w in ctx]
    c = task.ctrl
    seq = [c["bos"], *base, c["sep"], *(stoi[w] for w in gap), *base[:n_copy]]
    return seq, base[n_copy]


def make_batch(task: Task, n: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """右侧补 PAD 对齐。因果掩码保证真实 token 看不到后面的 PAD，补位安全；
    每个样本只取最后一个真实位置的 logits。"""
    ex = [sample_example(task, rng) for _ in range(n)]
    maxlen = max(len(e[0]) for e in ex)
    xs = torch.full((n, maxlen), task.ctrl["pad"], dtype=torch.long)
    last = torch.zeros(n, dtype=torch.long)
    ys = torch.zeros(n, dtype=torch.long)
    for i, (ids, y) in enumerate(ex):
        xs[i, : len(ids)] = torch.tensor(ids)
        last[i] = len(ids) - 1
        ys[i] = y
    return xs, ys, last


# ---------------------------------------------------------------- 模型

class Block(nn.Module):
    """Pre-LN Transformer block（GPT-2 之后的主流写法）。"""

    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d, dff = cfg.d_model, cfg.d_ff or 4 * cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, dff)
        self.w2 = nn.Linear(dff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 手写而不是调 SDPA，为的是每步中间张量都能拿出来看；与 SDPA 的差见 --check
        T = x.shape[1]
        h = self.ln1(x)
        q, k, v = (heads_of(t, T, self.cfg) for t in self.qkv(h).split(self.cfg.d_model, dim=2))
        probs = attn_probs(q, k, T, x.device, self.cfg.head_dim)
        mixed = (probs @ v).transpose(1, 2).reshape(*x.shape[:2], self.cfg.d_model)
        x = x + self.proj(mixed)                                  # 残差 1
        h = self.ln2(x)
        x = x + self.w2(F.gelu(self.w1(h)))                        # 残差 2
        return x


def heads_of(t: torch.Tensor, T: int, cfg: ModelCfg) -> torch.Tensor:
    """(B,T,d) -> (B,H,T,d/H)：把通道切成 H 份，每份独立做一次注意力（参数量不变）。"""
    return t.view(t.shape[0], T, cfg.n_heads, cfg.head_dim).transpose(1, 2)


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def attn_probs(q: torch.Tensor, k: torch.Tensor, T: int, device: torch.device, head_dim: int) -> torch.Tensor:
    """softmax(QKᵀ/√d + 因果掩码)。1/√d 缩放的作用见消融 score_std_*。"""
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    return scores.masked_fill(causal_mask(T, device), float("-inf")).softmax(-1)


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelCfg, task: Task) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(task.vocab, cfg.d_model)
        self.pos = nn.Embedding(task.max_t, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, task.vocab, bias=False)
        self.head.weight = self.emb.weight          # tied embedding（GPT-2/GPT-3 的做法）
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))               # (B,T,V)

    def params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def last_logits(model: TinyGPT, xs: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    return model(xs)[torch.arange(xs.shape[0]), last]


# ---------------------------------------------------------------- 训练 / 评测

def train(task: Task, cfg: ModelCfg, steps: int, seed: int, lr: float = 3e-3,
          batch: int = 128, verbose: bool = True) -> TinyGPT:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyGPT(cfg, task)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    model.train()
    logits = None
    for step in range(1, steps + 1):
        xs, y, last = make_batch(task, batch, rng)
        logits = last_logits(model, xs, last)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 250 == 0 or step == 1):
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"  step {step:5d}/{steps}  loss {loss.item():.4f}  acc {acc:.0%}")
    return model


@torch.no_grad()
def accuracy(model: TinyGPT, task: Task, n: int = 512, seed: int = 1234) -> float:
    model.eval()
    xs, y, last = make_batch(task, n, random.Random(seed))
    return (last_logits(model, xs, last).argmax(-1) == y).float().mean().item()


@torch.no_grad()
def accuracy_by_length(model: TinyGPT, task: Task, n_per: int = 120, seed: int = 7) -> dict:
    """按序列长度分组——验证它不是靠某个固定长度背下来的。"""
    model.eval()
    rng = random.Random(seed)
    buckets: dict[int, list[bool]] = {}
    for _ in range(n_per):
        ids, y = sample_example(task, rng)
        pred = model(torch.tensor([ids]))[0, -1].argmax().item()
        buckets.setdefault(len(ids), []).append(pred == y)
    return {str(k): round(sum(v) / len(v), 4) for k, v in sorted(buckets.items())}


# ---------------------------------------------------------------- 逐头诊断

@torch.no_grad()
def head_metrics(model: TinyGPT, task: Task, n: int = 120, seed: int = 5) -> dict:
    """每个头三个数：mean_shift / induction / 末行峰值位置。
    induction 分的定义：末位 token 的上一次出现位置 s，末行在 s+1 上的权重。
    这是判断"这个头到底在做什么"最直接的读数，也是页面里头的标签来源。"""
    model.eval()
    rng = random.Random(seed)
    cfg = model.cfg
    acc: dict[str, dict[str, list[float]]] = {}
    for _ in range(n):
        ids, _ = sample_example(task, rng)
        xs = torch.tensor([ids])
        T = len(ids)
        cur = ids[-1]
        prevs = [i for i in range(T - 1) if ids[i] == cur]
        s = prevs[-1] if prevs else None
        x = model.emb(xs) + model.pos(torch.arange(T))
        for li, blk in enumerate(model.blocks):
            q, k, _ = (heads_of(t, T, cfg) for t in blk.qkv(blk.ln1(x)).split(cfg.d_model, dim=2))
            probs = attn_probs(q, k, T, xs.device, cfg.head_dim)
            for hi in range(cfg.n_heads):
                key = f"L{li + 1}H{hi}"
                d = acc.setdefault(key, {"shift": [], "induction": [], "entropy": [], "peak": []})
                row = probs[0, hi]
                d["shift"].append(float(row.diagonal(offset=-1).mean()))
                d["induction"].append(float(row[-1, s + 1]) if s is not None else 0.0)
                d["entropy"].append(float(-(row.clamp_min(1e-9).log() * row).sum(-1).mean()))
                d["peak"].append(float(row[-1].max()))
            x = blk(x)

    def mean(v: list[float]) -> float:
        return round(sum(v) / len(v), 4) if v else 0.0

    return {k: {"mean_shift": mean(v["shift"]), "induction": mean(v["induction"]),
                "mean_entropy": mean(v["entropy"]), "mean_peak": mean(v["peak"])}
            for k, v in acc.items()}


# ---------------------------------------------------------------- 消融（无需重训）

def _zero_hook(module: nn.Module, args: tuple, out: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(out)


@torch.no_grad()
def ablation(model: TinyGPT, task: Task, n: int = 512, seed: int = 1234) -> dict:
    """逐步拆掉某一块，看准确率掉多少。掉得少 = 那一块在这个任务上没承重。"""
    model.eval()
    out: dict = {"baseline": accuracy(model, task, n, seed)}

    for li, blk in enumerate(model.blocks):
        for part, mod in (("attn", blk.proj), ("ffn", blk.w2)):
            h = mod.register_forward_hook(_zero_hook)
            out[f"zero_{part}_L{li + 1}"] = accuracy(model, task, n, seed)
            h.remove()

    # 扣掉位置嵌入 = "不告诉模型这是第几个词"
    class ZeroPos(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.d = d

        def forward(self, ids):  # noqa: ANN001
            return torch.zeros(ids.shape[0], self.d, device=ids.device)

    orig = model.pos
    model.pos = ZeroPos(model.cfg.d_model)  # type: ignore[assignment]
    out["no_pos_emb"] = accuracy(model, task, n, seed)
    model.pos = orig

    out.update(score_stats(model, task))
    return out


@torch.no_grad()
def score_stats(model: TinyGPT, task: Task) -> dict:
    """第一层打分统计：1/√d 缩放前后 softmax 的锐度，以及 Q/K 若共享权重会得到对称分数。"""
    model.eval()
    rng = random.Random(99)
    ids, _ = sample_example(task, rng)
    xs = torch.tensor([ids])
    T, cfg = len(ids), model.cfg
    h = model.blocks[0].ln1(model.emb(xs) + model.pos(torch.arange(T)))
    q, k, _ = (heads_of(t, T, cfg) for t in model.blocks[0].qkv(h).split(cfg.d_model, dim=2))
    raw = q @ k.transpose(-1, -2)
    mask = causal_mask(T, xs.device)
    # 对称性必须在掩码前算：填的是 -inf，两两相减会得到 inf 而不是真实差值
    asym = float((raw[0] - raw[0].transpose(-1, -2))[:, ~mask[0]].abs().max())
    raw = raw.masked_fill(mask, float("-inf"))
    scaled = raw / math.sqrt(cfg.head_dim)
    valid = ~mask

    out = {
        "seq_len": T,
        "score_std_scaled": round(float(scaled[:, :, valid].std()), 3),
        "score_std_unscaled": round(float(raw[:, :, valid].std()), 3),
        "attn_symmetry_max": round(asym, 3),
    }
    for tag, x in (("scaled", scaled), ("unscaled", raw)):
        p = x.softmax(-1).nan_to_num(0.0)
        out[f"max_prob_{tag}"] = round(float(p[:, :, valid].max()), 3)
        ent = -(p.clamp_min(1e-9).log() * p).sum(-1)     # 每行熵：0 = 只盯一个，log(T) = 平均分
        # 跳过第 0 行：BOS 只能看自己，熵恒为 0，会拉低均值
        out[f"entropy_{tag}"] = round(float(ent[0][:, 1:].mean()), 3)
    return out


# ---------------------------------------------------------------- 追踪

def ri(x: torch.Tensor | float) -> list:
    return [round(float(v), 3) for v in torch.as_tensor(x, dtype=torch.float32).flatten()]


def ri4(x: torch.Tensor) -> list:
    return [round(float(v), 4) for v in torch.as_tensor(x, dtype=torch.float32).flatten()]


@torch.no_grad()
def trace(model: TinyGPT, task: Task, words: list[str]) -> dict:
    """跑一次完整前向，把每一层的中间张量都录下来。"""
    model.eval()
    stoi, cfg = task.stoi, model.cfg
    if len(words) > task.max_ctx:
        raise ValueError(f"prompt 最多 {task.max_ctx} 个词，收到 {len(words)}")
    n_copy = 2                        # 重复前 2 个词，答案落在 SEP(+gap) 之后的第 3 位
    base = [stoi[w] for w in words]
    c = task.ctrl
    ids = [c["bos"], *base, c["sep"], *base[:n_copy]]
    T = len(ids)
    tokens = ["<BOS>", *words, "<SEP>", *words[:n_copy]]
    xs = torch.tensor([ids])
    target_id, target_word = base[n_copy], words[n_copy]

    x0 = model.emb(xs) + model.pos(torch.arange(T))
    steps: dict = {
        "tokens": tokens,
        "ids": ids,
        "n_copy": n_copy,
        "embedding_rows": ri(model.emb.weight[ids]),
        "pos_rows": ri(model.pos.weight[:T]),
        "embedding_norms": [round(float(v), 2) for v in model.emb.weight[ids].norm(dim=-1)],
        "x0": ri(x0),
        "target_word": target_word,
        "target_id": target_id,
        "target_first_pos": 1 + n_copy,     # 答案词在第一次出现时的位置（BOS 占 0）
        "last_pos": T - 1,
        "query_pos": T - 1,
    }

    layers = []
    x = x0
    for blk in model.blocks:
        h1 = blk.ln1(x)
        qkv = blk.qkv(h1)
        q, k, v = qkv.split(cfg.d_model, dim=2)
        qh, kh, vh = (heads_of(t, T, cfg) for t in (q, k, v))
        probs = attn_probs(qh, kh, T, xs.device, cfg.head_dim)
        mixed = (probs @ vh).transpose(1, 2).reshape(1, T, cfg.d_model)
        attn_out = blk.proj(mixed)
        x1 = x + attn_out
        h2 = blk.ln2(x1)
        ffn_pre = blk.w1(h2)
        ffn_act = F.gelu(ffn_pre)
        ffn_out = blk.w2(ffn_act)
        x2 = x1 + ffn_out

        layers.append({
            "h1": ri(h1), "h1_mean": ri4(h1.mean(-1)), "h1_std": ri4(h1.std(-1)),
            "q": ri(q), "k": ri(k), "v": ri(v),
            "scores": ri(qh @ kh.transpose(-1, -2) / math.sqrt(cfg.head_dim)),
            "probs": ri4(probs),
            "attn_mixed": ri(mixed),
            "attn_out": ri(attn_out),
            "attn_out_norm": [round(float(n), 2) for n in attn_out[0].norm(dim=-1)],
            "x_after_attn": ri(x1),
            "ffn_pre": ri(ffn_pre),
            "ffn_act": ri(ffn_act),
            "ffn_out": ri(ffn_out),
            "ffn_out_norm": [round(float(n), 2) for n in ffn_out[0].norm(dim=-1)],
            "ffn_active_frac": [round(float((a > 1e-4).float().mean()), 3) for a in ffn_act[0]],
            "x_out": ri(x2),
            "residual_ratio": [round(float(attn_out[0, t].norm() / max(1e-9, x[0, t].norm())), 3)
                               for t in range(T)],
            # 这一层的注意力输出，在查询位置上与"答案词嵌入"的方向一致性。
            # 高 = 这一层真的把答案词的内容搬到了查询位置（copy 的直接证据）。
            "cos_mixed_target": round(float(F.cosine_similarity(
                attn_out[0, -1].unsqueeze(0), model.emb.weight[target_id].unsqueeze(0)).item()), 3),
        })
        x = x2
    steps["layers"] = layers

    hf = model.ln_f(x)
    logits = model.head(hf)[0, -1]
    probs_all = logits.softmax(-1)
    top = torch.topk(probs_all, 8)
    steps["head"] = {
        "x_final": ri(x[0, -1]),
        "lnf_mean": round(float(hf[0, -1].mean()), 4),
        "lnf_std": round(float(hf[0, -1].std()), 4),
        "logits": ri(logits),
        "top_ids": [int(i) for i in top.indices],
        "top_words": [task.tokens[int(i)] for i in top.indices],
        "top_probs": [round(float(p), 4) for p in top.values],
        "entropy": round(float(-(probs_all.clamp_min(1e-12).log() * probs_all).sum()), 4),
    }
    steps["dla"] = direct_logit_attribution(model, steps, target_id, x[:, -1])
    # token 级归因：目标 token 是 base[n_copy]（与 trace 的 target 一致）
    steps["token_attr"] = token_attribution(model, task, ids)
    # 把每层注意力的混合输出（(1,T,d) 展平）也存下来，页面用它做"copy 到哪"的解释
    return {"prompt_words": words, "seq_len": T, "steps": steps,
            "head_metrics": head_metrics(model, task)}


@torch.no_grad()
def direct_logit_attribution(model: TinyGPT, steps: dict, target_id: int,
                             x_final_pos: torch.Tensor) -> dict:
    """直接 logit 归因：把答案词的最终 logit 拆到各个部件头上。

每个部件把 x 加进残差流，所以 x_final = x0 + Σ 部件输出。最终 logit 是
W_U[target] · LN(x_final)。LN 带仿射参数 (g, b)，在 x_final 处把 LN 线性化：

    LN(x) ≈ g ⊙ (x − mean(x)) / std(x) + b

于是把每个部件的输出投影到"答案词的方向"上，就得到它对这个 logit 的贡献。
（**线性化**归因，不是 Shapley 值；但残差流的加法性让它足够准确，且能逐项对上号。）
    """
    cfg = model.cfg
    x_final = x_final_pos.float()[None, :]
    hf = model.ln_f(x_final)[0]
    w = model.head.weight[target_id]           # 答案词的 unembed 方向
    T, d = len(steps["tokens"]), cfg.d_model

    def last(tag_rows: list) -> torch.Tensor:
        """ri() 把 (1,T,d) 展平了；还原成矩阵再取最后一位置。"""
        return torch.tensor(tag_rows, dtype=torch.float32).reshape(T, d)[-1]

    comps = {"embed+pos": last(steps["x0"])}
    for li, L in enumerate(steps["layers"], start=1):
        comps[f"attn_L{li}"] = last(L["attn_out"])
        comps[f"ffn_L{li}"] = last(L["ffn_out"])

    base_logit = float((w * hf).sum())

    def logit_without(name: str) -> float:
        """把这一块从 x_final 里扣掉，重算 logit——**精确**的留一贡献，不用线性化。"""
        y = model.ln_f((x_final - comps[name])[0]).flatten()
        return float((w @ y).item())

    contrib = {name: base_logit - logit_without(name) for name in comps}
    total = sum(contrib.values())
    return {
        "contrib": {k: round(v, 3) for k, v in contrib.items()},
        "total": round(total, 3),
        "actual_logit": round(base_logit, 3),
        "share": {k: round(v / total, 2) if total else 0.0 for k, v in contrib.items()},
        # 被扣掉后 logit 降到多少（页面用它显示"没有这一块，答案词的分数会变成多少"）
        "logit_without": {k: round(logit_without(k), 3) for k in comps},
    }


@torch.no_grad()
def token_attribution(model: TinyGPT, task: Task, ids: list[int], n_sources: int = 4) -> list[dict]:
    """把答案词的 logit 归到**输入 token**上，用"扣掉它再算一遍"的精确留一法。

    做法：在第 2 层注意力里，把"第 src 个位置作为 V 源"的那一份权重清零并重新归一化，
    再跑完整前向，看答案词的 logit 掉多少。掉得越多，说明这个 token 越是答案的来源。

    这比看注意力权重更硬：注意力权重只说"看多远"，这里是"不看它之后分数变了多少"。
    实现上必须作用在 **p@V 那一步**，而不是 proj 的输出：
      attn_out = W_O(concat_heads(Σ_s p[s]·V[s]))
    若从 proj 输出里扣，扣掉的已是 W_O 混合后的向量，而且 out[0,src] 是"src 作为
    query 的结果"，不是"src 作为被读的源"，语义不对。
    """
    model.eval()
    base_ids = list(ids)
    T = len(base_ids)
    xs = torch.tensor([base_ids])

    cur = base_ids[-1]
    prevs = [i for i in range(T - 1) if base_ids[i] == cur]
    s = prevs[-1] if prevs else None
    answer = base_ids[s + 1] if s is not None and s + 1 < T - 1 else None
    if answer is None:
        return []
    base = float(model(xs)[0, -1, answer].item())

    cfg, blk = model.cfg, model.blocks[-1]
    # 预先算好第 2 层输入处的 probs 与 V（与真前向完全一致）
    x_in = model.blocks[0](model.emb(xs) + model.pos(torch.arange(T)))
    h = blk.ln1(x_in)
    q, k, v = (heads_of(t, T, cfg) for t in blk.qkv(h).split(cfg.d_model, dim=2))
    p = attn_probs(q, k, T, xs.device, cfg.head_dim)

    rows = []
    for src in range(T - 1):
        p2 = p.clone()
        p2[:, :, :, src] = 0.0
        p2 = p2 / p2.sum(-1, keepdim=True).clamp_min(1e-9)
        mixed = (p2 @ v).transpose(1, 2).reshape(1, T, cfg.d_model)
        x2 = x_in + blk.proj(mixed)
        x2 = x2 + blk.w2(F.gelu(blk.w1(blk.ln2(x2))))
        logits = model.head(model.ln_f(x2))[0, -1]
        rows.append({"pos": src, "token": task.tokens[base_ids[src]],
                     "logit_drop": round(base - float(logits[answer]), 2)})
    rows.sort(key=lambda r: -r["logit_drop"])
    return rows[:n_sources]


def _prompt_ids(task: Task, words: list[str]) -> tuple[list[int], list[str]]:
    stoi, c = task.stoi, task.ctrl
    base = [stoi[w] for w in words]
    return ([c["bos"], *base, c["sep"], *base[:2]],
            ["<BOS>", *words, "<SEP>", *words[:2]])


@torch.no_grad()
def _probs_and_scores(model: TinyGPT, task: Task, words: list[str]) -> list[dict]:
    """只为"未训练对照"取两张矩阵：每层的 attention probs 与原始打分。"""
    model.eval()
    ids, _ = _prompt_ids(task, words)
    xs = torch.tensor([ids])
    T, cfg = len(ids), model.cfg
    x = model.emb(xs) + model.pos(torch.arange(T))
    out = []
    for blk in model.blocks:
        q, k, _ = (heads_of(t, T, cfg) for t in blk.qkv(blk.ln1(x)).split(cfg.d_model, dim=2))
        raw = q @ k.transpose(-1, -2) / math.sqrt(cfg.head_dim)
        out.append({"probs": ri4(attn_probs(q, k, T, xs.device, cfg.head_dim)), "scores": ri(raw)})
        x = blk(x)
    return out


@torch.no_grad()
def _head_probs(model: TinyGPT, task: Task, words: list[str]) -> dict:
    model.eval()
    ids, _ = _prompt_ids(task, words)
    logits = model(torch.tensor([ids]))[0, -1]
    p = logits.softmax(-1)
    top = torch.topk(p, 6)
    return {"top_words": [task.tokens[int(i)] for i in top.indices],
            "top_probs": [round(float(v), 4) for v in top.values],
            "entropy": round(float(-(p.clamp_min(1e-12).log() * p).sum()), 4)}


# ---------------------------------------------------------------- 自检

def self_check(task: Task = TASK, cfg: ModelCfg = CFG) -> None:
    torch.manual_seed(0)
    model = TinyGPT(cfg, task)
    n, T = 2, 9
    xs = torch.randint(0, task.vocab, (n, T))
    b = model.blocks[0]
    h = b.ln1(model.emb(xs) + model.pos(torch.arange(T)))
    q, k, v = (heads_of(t, T, cfg) for t in b.qkv(h).split(cfg.d_model, dim=2))
    ref = b.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True)
                 .transpose(1, 2).reshape(n, T, cfg.d_model))
    manual = b.proj((attn_probs(q, k, T, xs.device, cfg.head_dim) @ v)
                    .transpose(1, 2).reshape(n, T, cfg.d_model))
    diff = float((manual - ref).detach().abs().max())
    print(f"  [1] 手写注意力 vs SDPA 最大差: {diff:.2e}  (应 < 1e-5)")
    assert diff < 1e-5, "注意力实现与 SDPA 不一致"

    print(f"  [2] 未训练模型准确率: {accuracy(model, task, n=256):.0%}  (随机基线 ≈ {1 / len(task.words):.0%})")
    probs = attn_probs(q, k, T, xs.device, cfg.head_dim)
    print(f"  [3] 每行 attention 之和 = 1: {float(probs.sum(-1).min()):.4f} .. {float(probs.sum(-1).max()):.4f}")

    rng = random.Random(0)
    lens = [len(sample_example(task, rng)[0]) for _ in range(200)]
    print(f"  [4] 序列长度覆盖: {sorted(set(lens))}  (可变，堵死绝对位置捷径)")
    assert len(set(lens)) >= 3, "长度不含变化，模型会背位置"
    assert max(lens) <= task.max_t, f"最长样本 {max(lens)} 超过位置表 {task.max_t}"

    print(f"  [5] tied embedding：head.weight 就是 emb.weight → {model.head.weight is model.emb.weight}")
    print(f"  [6] 参数量 {model.params():,}  (d={cfg.d_model}, H={cfg.n_heads}, L={cfg.n_layers}, V={task.vocab})")


# ---------------------------------------------------------------- 入口

SCAN_CONFIGS = [
    ModelCfg(d_model=32, n_heads=4, n_layers=2),
    ModelCfg(d_model=64, n_heads=4, n_layers=2),
    ModelCfg(d_model=64, n_heads=8, n_layers=2),
]


def scan(task: Task, steps: int, seed: int) -> None:
    """扫描几组配置，报告「准确率 / 消融 / 每头 induction 分」。
    选配置的标准不是准确率（都能 100%），而是**第 2 层注意力是否承重**：
    zero_attn_L2 越低说明 induction 电路越是真的在用。"""
    print(f"{'cfg':22s} {'acc':>5s} {'L1attn':>7s} {'L2attn':>7s} {'L1ffn':>6s} {'L2ffn':>6s} "
          f"{'pos':>5s} {'best_ind':>20s}")
    for cfg in SCAN_CONFIGS:
        m = train(task, cfg, steps, seed, verbose=False)
        a = ablation(m, task, n=400)
        hm = head_metrics(m, task, n=100)
        best = max(hm.items(), key=lambda kv: kv[1]["induction"])
        print(f"d{cfg.d_model}/H{cfg.n_heads}/L{cfg.n_layers:<10}"
              f" {a['baseline']:5.2f} {a['zero_attn_L1']:7.2f} {a['zero_attn_L2']:7.2f}"
              f" {a['zero_ffn_L1']:6.2f} {a['zero_ffn_L2']:6.2f} {a['no_pos_emb']:5.2f}"
              f"   {best[0]} {best[1]['induction']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=CFG.d_model)
    ap.add_argument("--heads", type=int, default=CFG.n_heads)
    ap.add_argument("--layers", type=int, default=CFG.n_layers)
    ap.add_argument("--check", action="store_true", help="只跑自检")
    ap.add_argument("--scan", action="store_true", help="扫描配置")
    ap.add_argument("--out", default="docs/transformer-trace.json")
    ap.add_argument("--prompt", default="猫 追 蝴蝶 穿过 花园 藏进")
    args = ap.parse_args()

    cfg = ModelCfg(d_model=args.d_model, n_heads=args.heads, n_layers=args.layers)
    task = TASK
    if cfg.d_model % cfg.n_heads != 0:
        raise SystemExit(f"d_model={cfg.d_model} 不能被 n_heads={cfg.n_heads} 整除")

    print("自检:")
    self_check(task, cfg)
    if args.check:
        return
    if args.scan:
        scan(task, args.steps, args.seed)
        return

    print(f"训练 {args.steps} 步:")
    model = train(task, cfg, args.steps, args.seed)
    acc = accuracy(model, task)
    print(f"最终准确率: {acc:.1%}  (512 个随机样本)")

    print("消融:")
    abl = ablation(model, task)
    for k, v in abl.items():
        print(f"  {k:22s} {v}")
    print("逐头诊断 (induction = 末位盯住'上次出现位置+1'的权重):")
    hm = head_metrics(model, task)
    for k, v in sorted(hm.items(), key=lambda kv: -kv[1]["induction"])[:6]:
        print(f"  {k}  induction={v['induction']:.2f}  shift={v['mean_shift']:.2f}  "
              f"entropy={v['mean_entropy']:.2f}  peak={v['mean_peak']:.2f}")

    tr = trace(model, task, args.prompt.split())
    tr["model"] = {"d_model": cfg.d_model, "n_heads": cfg.n_heads, "n_layers": cfg.n_layers,
                   "d_ff": cfg.d_ff or 4 * cfg.d_model, "vocab": task.vocab,
                   "max_t": task.max_t, "head_dim": cfg.head_dim, "params": model.params()}
    tr["ablation"] = abl
    tr["accuracy"] = round(acc, 4)
    tr["accuracy_by_length"] = accuracy_by_length(model, task)
    tr["task"] = {"min_ctx": task.min_ctx, "max_ctx": task.max_ctx,
                  "max_gap": task.max_gap, "vocab_words": list(task.words),
                  "tokens": list(task.tokens)}
    tr["train"] = {"steps": args.steps, "seed": args.seed, "lr": 3e-3, "batch": 128}
    # 未训练的同架构对照：同一套矩阵，一张是"随机"，一张是"学成"。
    # 页面用它来展示"训练到底改了什么"——结构不变，权重变了。
    torch.manual_seed(args.seed + 999)
    untrained = TinyGPT(cfg, task).eval()
    tr["untrained"] = {
        "layers": _probs_and_scores(untrained, task, args.prompt.split()),
        "head": _head_probs(untrained, task, args.prompt.split()),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    js = ROOT / "web" / "transformer-trace.js"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(
        "// 由 scripts/tiny_transformer.py 生成，勿手改。\n"
        "window.TRANSFORMER_TRACE = " + json.dumps(tr, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"写出 {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"写出 {js.relative_to(ROOT)} ({js.stat().st_size / 1024:.0f} KB)")
    print("按长度分组准确率:", tr["accuracy_by_length"])
    h = tr["steps"]["head"]
    print(f"预测: {' | '.join(f'{w} {p:.1%}' for w, p in zip(h['top_words'], h['top_probs']))}")
    print(f"正确答案: {tr['steps']['target_word']}")


if __name__ == "__main__":
    main()
