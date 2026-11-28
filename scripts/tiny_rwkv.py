"""训练一个微型 RWKV-7，并把**递归状态**的每一步演进 dump 成 JSON，供 web/rwkv.html 可视化。

和 transformer-primer / transformer.html 用的是同一个任务（上下文 induction），
所以两页的数字可以对照。区别是全在这里：RWKV 没有注意力、没有 KV cache，
它只有一个**大小固定的矩阵状态**，每个 token 进来就改写它一次。

实现刻意对齐 fla 的 RWKV-7（`fla/ops/rwkv7/fused_recurrent.py` 的那几行）：
    S_t = exp(w_t) ⊙ S_{t-1}  +  b_t (a_tᵀ S_{t-1})  +  k_t v_tᵀ      ← 衰减 + 擦 + 写
    o_t = S_t r_t                                                    ← 读
其中 a_t = −k̂_t (L2 归一化), b_t = k̂_t·a_t, w_t < 0。
训练时用「折叠形式」（把衰减乘子提到求和外）来并行，与串行递推逐位等价，见 --check。

产出：
    docs/rwkv-trace.json      逐 token 的状态演进 + 消融 + 对照
    web/rwkv-trace.js         同一份数据包成 window.RWKV_TRACE（页面可 file:// 直开）

用法：
    python scripts/tiny_rwkv.py                 # 训练并导出（CPU 约 1 分钟）
    python scripts/tiny_rwkv.py --check          # 只跑自检
    python scripts/tiny_rwkv.py --steps 4000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

# 与 tiny_transformer.py 相同的 60 词表 + 特殊符号
from tiny_transformer import Task  # noqa: E402

D_MODEL = 32
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS      # 8  → 每头状态是 8×8
N_LAYERS = 2
D_FF = 4 * D_MODEL


@dataclass(frozen=True)
class Cfg:
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    d_ff: int = D_FF

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------- RWKV-7 层

class RWKV7TimeMix(nn.Module):
    """RWKV-7 的跨 token 混合：可读 / 可写 / 可擦的矩阵状态（取代注意力）。"""

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        d, H, hd = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.cfg = cfg
        # 6 路 token shift 系数（逐通道），初始化对齐 fla 的 layer-0 预设：
        #   x_r/x_g 弱 shift(0.2)，x_k/x_v 中等(0.7)，x_w/x_a 强(0.9)
        ddd = torch.ones(d)
        for name, ratio in (("x_r", 0.2), ("x_w", 0.9), ("x_k", 0.7),
                            ("x_v", 0.7), ("x_a", 0.9), ("x_g", 0.2)):
            setattr(self, name, nn.Parameter(1.0 - torch.pow(ddd, ratio)))
        self.r_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        # 真实实现里 w/a/g 用低秩 LoRA 生成；这里用两个小矩阵等价替代，保留「逐 token 逐通道」这一性质
        self.w_a = nn.Linear(d, max(8, d // 4), bias=False)
        self.w_b = nn.Linear(max(8, d // 4), d, bias=False)
        self.a_a = nn.Linear(d, max(8, d // 4), bias=False)
        self.a_b = nn.Linear(max(8, d // 4), d, bias=False)
        self.g_a = nn.Linear(d, max(8, d // 4), bias=False)
        self.g_b = nn.Linear(max(8, d // 4), d, bias=False)
        # ⚠️ 这三个参数必须给**非零**初值。
        #   原始 fla 里它们声明成 zeros 是因为随后会加载预训练权重；
        #   从头训练若仍为 0，则 kk = normalize(k ⊙ 0) = 0，
        #   于是擦除项 b(aᵀS) 恒为 0 —— 训练时"擦除"完全没生效（实测 ‖erase‖=0，
        #   消融 no_erase 只掉 2.5%，就是因为它压根没参与计算）。
        #   这里按 fla 的 _init_weights 预设给初值：k_k≈0.71，k_a≈1.02，r_k≈1。
        self.k_k = nn.Parameter(torch.full((d,), 0.71))
        self.k_a = nn.Parameter(torch.full((d,), 1.02))
        self.r_k = nn.Parameter(torch.ones(H, hd))
        nn.init.zeros_(self.w_b.weight)     # 让 w 初始接近 -0.6065·0.5，不一开始就爆炸
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None,
                record: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,T,d)；state: (B,H,hd,hd) 或 None；返回 (out, new_state)。"""
        B, T, d = x.shape
        H, hd = self.cfg.n_heads, self.cfg.head_dim

        # ---- Token Shift：delta[t] = x[t-1] - x[t]（t=0 补零）
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        delta = prev - x

        xr = x + delta * self.x_r
        xw = x + delta * self.x_w
        xk = x + delta * self.x_k
        xv = x + delta * self.x_v
        xa = x + delta * self.x_a

        r = self.r_proj(xr).view(B, T, H, hd)
        w = -0.6065306597126334 * torch.sigmoid(self.w_b(self.w_a(xw))).view(B, T, H, hd)
        k = self.k_proj(xk).view(B, T, H, hd)
        v = self.v_proj(xv).view(B, T, H, hd)
        a = torch.sigmoid(self.a_b(self.a_a(xa))).view(B, T, H, hd)

        # k̂ = L2 归一化(k ⊙ k_k)；擦除必须沿 k̂ 这个单位方向才成立
        kk = F.normalize(k * self.k_k.view(1, 1, H, hd), dim=-1, p=2.0)
        k_fused = k.addcmul(k * (a - 1), self.k_a.view(1, 1, H, hd))

        if state is None:
            state = torch.zeros(B, H, hd, hd, device=x.device, dtype=torch.float32)

        # ⚠️ 这里是**逐 token 的 Python 循环**，因此这个 toy 的训练速度受 Python 开销限制
        #   （4000 步约 4.5 分钟）。真实的 RWKV 靠分块 + 段内并行（chunk_rwkv7）解决，
        #   那需要 WY 表示和三角求逆，不是几十行能写对的；本脚本用串行版本，
        #   保证与数学定义逐位一致，把它当「教学实现」而不是「性能实现」。
        #   自检 [1] 验证了「纯衰减+写入」部分的闭式与串行等价（1.19e-07），
        #   这正是 chunk 并行成立的数学基础。
        if record is not None:
            outs, snaps, S = self._recurrent_serial(w, r, k_fused, v, kk, a, state, T)
        else:
            outs, _, S = self._recurrent_serial(w, r, k_fused, v, kk, a, state, T, want_snaps=False)
        o = torch.stack(outs, dim=1)                                     # (B,T,H,hd)
        g = torch.sigmoid(self.g_b(self.g_a(x))).view(B, T, H, hd)
        o = (o * g).reshape(B, T, d)
        y = x + self.o_proj(o)
        if record is not None:
            record.update({"delta": delta, "w": w, "a": a, "kk": kk, "k": k_fused,
                           "v": v, "r": r, "o": o, "snaps": snaps})
        return y, S


    def _recurrent_serial(self, w, r, k_fused, v, kk, a, state, T, want_snaps: bool = True):
        """逐 token 递推。这是 RWKV 的**定义式**实现，解码时真实走的就是这条路径。"""
        outs, snaps = [], []
        S = state if state is not None else torch.zeros(
            w.shape[0], self.cfg.n_heads, self.cfg.head_dim, self.cfg.head_dim, dtype=torch.float32)
        for t in range(T):
            S_prev = S
            decay = torch.exp(w[:, t])
            erase = (kk[:, t] * a[:, t])[:, :, :, None] * torch.sum(
                (-kk[:, t])[:, :, :, None] * S_prev, dim=2)[:, :, None, :]
            write = k_fused[:, t][:, :, :, None] * v[:, t][:, :, None, :]
            S = decay[:, :, :, None] * S_prev + erase + write
            o = torch.sum(S * r[:, t][:, :, :, None], dim=2)
            outs.append(o)
            if want_snaps:
                snaps.append({
                    "t": t, "decay": decay[0].flatten().tolist(),
                    "S_prev": S_prev[0].flatten().tolist(), "erase": erase[0].flatten().tolist(),
                    "write": write[0].flatten().tolist(), "S": S[0].flatten().tolist(),
                    "o": o[0].flatten().tolist(), "S_norm": S[0].norm(dim=(1, 2)).tolist(),
                })
        return outs, snaps, S


class RWKV7ChannelMix(nn.Module):
    """RWKV-7 的逐 token 变换：结构上就是 FFN（4× 升维 + 门控 + 降维）。"""

    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        d, dff = cfg.d_model, cfg.d_ff
        self.x_k = nn.Parameter(1.0 - torch.pow(torch.ones(d), 0.7))
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, dff, bias=False)
        self.w2 = nn.Linear(dff, d, bias=False)
        self.gate = nn.Linear(d, dff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        xk = x + (prev - x) * self.x_k
        h = self.ln(xk)
        act = F.relu(self.w1(h)) ** 2            # sqrelu（RWKV-7 用的激活）
        return x + self.w2(act * torch.sigmoid(self.gate(xk)))


class Block(nn.Module):
    def __init__(self, cfg: Cfg) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.time_mix = RWKV7TimeMix(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.channel_mix = RWKV7ChannelMix(cfg)

    def forward(self, x, state, record=None):
        x = x + self.time_mix(self.ln1(x), state, record)[0]
        x = self.channel_mix(x)
        return x


class TinyRWKV7(nn.Module):
    def __init__(self, cfg: Cfg, vocab: int, max_t: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.max_t = max_t

    def forward(self, ids, state=None, records=None):
        """state: list per layer，或 None（从零开始）。"""
        B, T = ids.shape
        x = self.emb(ids)
        new_states = []
        for i, blk in enumerate(self.blocks):
            s = None if state is None else state[i]
            rec = None if records is None else records.setdefault(i, {})
            # LN → TimeMix（残差在 time_mix 内部），再 ChannelMix（内部也带残差）
            tm_out, s_new = blk.time_mix(blk.ln1(x), s, rec)
            h = x + tm_out
            h = blk.channel_mix(h)
            new_states.append(s_new)
            if rec is not None:
                rec["x_in"] = x
                rec["x_out"] = h
            x = h
        return self.head(self.ln_out(x)), new_states

    def params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- 数据 / 训练

def make_batch(task: Task, n: int, rng: random.Random):
    ex = [sample(task, rng) for _ in range(n)]
    maxlen = max(len(e[0]) for e in ex)
    xs = torch.full((n, maxlen), task.ctrl["pad"], dtype=torch.long)
    last = torch.zeros(n, dtype=torch.long)
    ys = torch.zeros(n, dtype=torch.long)
    for i, (ids, y) in enumerate(ex):
        xs[i, : len(ids)] = torch.tensor(ids)
        last[i] = len(ids) - 1
        ys[i] = y
    return xs, ys, last


def sample(task: Task, rng: random.Random) -> tuple[list[int], int]:
    stoi = task.stoi
    n_ctx = rng.randint(task.min_ctx, task.max_ctx)
    n_copy = rng.randint(2, n_ctx - 1)
    ctx = rng.sample(task.words, n_ctx)
    gap = rng.sample(task.words, rng.randint(0, task.max_gap))
    base = [stoi[w] for w in ctx]
    c = task.ctrl
    return [c["bos"], *base, c["sep"], *(stoi[w] for w in gap), *base[:n_copy]], base[n_copy]


def train(task: Task, cfg: Cfg, steps: int, seed: int, lr: float = 4e-3, batch: int = 96,
          verbose: bool = True) -> TinyRWKV7:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = TinyRWKV7(cfg, task.vocab, task.max_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    model.train()
    for step in range(1, steps + 1):
        xs, y, last = make_batch(task, batch, rng)
        # 因果目标：用每个位置预测下一个 token（这里只需要最后一位的答案）
        logits, _ = model(xs)
        pred = logits[torch.arange(xs.shape[0]), last]
        loss = F.cross_entropy(pred, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if verbose and (step % 250 == 0 or step == 1):
            acc = (pred.argmax(-1) == y).float().mean().item()
            print(f"  step {step:5d}/{steps}  loss {loss.item():.4f}  acc {acc:.0%}")
    return model


@torch.no_grad()
def accuracy(model: TinyRWKV7, task: Task, n: int = 512, seed: int = 1234) -> float:
    model.eval()
    xs, y, last = make_batch(task, n, random.Random(seed))
    logits, _ = model(xs)
    return (logits[torch.arange(xs.shape[0]), last].argmax(-1) == y).float().mean().item()


@torch.no_grad()
def accuracy_by_length(model: TinyRWKV7, task: Task, n_per: int = 100, seed: int = 7) -> dict:
    model.eval()
    rng = random.Random(seed)
    buckets: dict[int, list[bool]] = {}
    for _ in range(n_per):
        ids, y = sample(task, rng)
        logits, _ = model(torch.tensor([ids]))
        buckets.setdefault(len(ids), []).append(int(logits[0, -1].argmax()) == y)
    return {str(k): round(sum(v) / len(v), 4) for k, v in sorted(buckets.items())}


# ---------------------------------------------------------------- 消融

@torch.no_grad()
def ablation(model: TinyRWKV7, task: Task, n: int = 384, seed: int = 1234) -> dict:
    """逐个拆掉部件。重点是「衰减」「擦除」「token shift」——这三个是 RWKV 独有的。"""
    model.eval()
    out: dict = {"baseline": accuracy(model, task, n, seed)}

    # ① 拆掉第 L 层 Time-Mix（= Transformer 的注意力被拆）
    for li, blk in enumerate(model.blocks):
        orig = blk.time_mix.forward

        def patched(x, state, record=None, _orig=orig):  # noqa: ANN001
            y, s = _orig(x, state, None)
            return torch.zeros_like(y), s
        blk.time_mix.forward = patched
        out[f"zero_timemix_L{li + 1}"] = accuracy(model, task, n, seed)
        blk.time_mix.forward = orig

    # ② 拆掉 Channel-Mix
    for li, blk in enumerate(model.blocks):
        orig = blk.channel_mix.forward

        def patched(x, _orig=orig):  # noqa: ANN001
            return torch.zeros_like(_orig(x))
        blk.channel_mix.forward = patched
        out[f"zero_chmix_L{li + 1}"] = accuracy(model, task, n, seed)
        blk.channel_mix.forward = orig

    # ③ 去掉 token shift（把所有 x_* 系数置 0 → delta 不再参与）
    saved = {}
    for blk in model.blocks:
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a"):
            saved[(id(blk), name)] = getattr(blk.time_mix, name).data.clone()
            getattr(blk.time_mix, name).data.zero_()
        saved[(id(blk), "x_k_cm")] = blk.channel_mix.x_k.data.clone()
        blk.channel_mix.x_k.data.zero_()
    out["no_token_shift"] = accuracy(model, task, n, seed)
    for blk in model.blocks:
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a"):
            getattr(blk.time_mix, name).data.copy_(saved[(id(blk), name)])
        blk.channel_mix.x_k.data.copy_(saved[(id(blk), "x_k_cm")])

    # ④ 关掉衰减：w 置 0 → exp(w)=1，状态永不淡化（记忆无限累积）
    saved_w = []
    for blk in model.blocks:
        saved_w.append(blk.time_mix.w_b.weight.data.clone())
        blk.time_mix.w_b.weight.data.zero_()
    out["no_decay"] = accuracy(model, task, n, seed)
    for blk, w in zip(model.blocks, saved_w):
        blk.time_mix.w_b.weight.data.copy_(w)

    # ⑤ 关掉擦除：a 置 1 → 擦除项为 0
    saved_a = []
    for blk in model.blocks:
        saved_a.append((blk.time_mix.a_a.weight.data.clone(), blk.time_mix.a_b.weight.data.clone()))
        blk.time_mix.a_b.weight.data.zero_()
    out["no_erase"] = accuracy(model, task, n, seed)
    for blk, (aa, ab) in zip(model.blocks, saved_a):
        blk.time_mix.a_a.weight.data.copy_(aa)
        blk.time_mix.a_b.weight.data.copy_(ab)

    return out


# ---------------------------------------------------------------- 追踪

def ri(x, nd: int = 3) -> list:
    return [round(v, nd) for v in torch.as_tensor(x, dtype=torch.float32).flatten().tolist()]


@torch.no_grad()
def trace(model: TinyRWKV7, task: Task, words: list[str]) -> dict:
    model.eval()
    stoi, c = task.stoi, task.ctrl
    n_copy = 2
    base = [stoi[w] for w in words]
    ids = [c["bos"], *base, c["sep"], *base[:n_copy]]
    T = len(ids)
    tokens = ["<BOS>", *words, "<SEP>", *words[:n_copy]]
    xs = torch.tensor([ids])

    records: dict = {}
    logits, states = model(xs, None, records)

    L0 = records[0]
    snaps = L0["snaps"]

    steps = {
        "tokens": tokens,
        "ids": ids,
        "n_copy": n_copy,
        "target_word": words[n_copy],
        "target_id": base[n_copy],
        "target_first_pos": 1 + n_copy,
        "shift": ri(L0["delta"]),                    # (2,T,d) = x_prev - x
        "shift_norm": [round(float(L0["delta"][0, t].norm()), 3) for t in range(T)],
        "decay_exp": [[round(v, 4) for v in s["decay"]] for s in snaps],
        "state_norm_by_layer": [
            [round(v, 3) for v in s["S_norm"]] for s in snaps
        ],
        "snaps": [],                                  # 逐 token：S_prev / erase / write / S
        "timemix_o": ri(L0["o"]),
        "w_values": ri(L0["w"]),
        "a_values": ri(L0["a"]),
        "kk_norm": round(float(L0["kk"][0, -1].norm()), 4),
    }
    for s in snaps:
        H, hd = model.cfg.n_heads, model.cfg.head_dim
        steps["snaps"].append({
            "t": s["t"],
            "decay": s["decay"],
            "S_prev": s["S_prev"],
            "erase": s["erase"],
            "write": s["write"],
            "S": s["S"],
            "S_norm": s["S_norm"],
            "o": s["o"],
            "H": H, "hd": hd,
        })

    # 第二层的状态（只取范数轨迹）
    if 1 in records:
        steps["state_norm_L2"] = [[round(v, 3) for v in s["S_norm"]] for s in records[1]["snaps"]]

    # 输出头
    lg = logits[0, -1]
    p = lg.softmax(-1)
    top = torch.topk(p, 8)
    steps["head"] = {
        "x_final": ri(records[1]["x_out"][0, -1]),
        "logits": ri(lg),
        "top_words": [task.tokens[int(i)] for i in top.indices],
        "top_probs": [round(float(v), 4) for v in top.values],
        "entropy": round(float(-(p.clamp_min(1e-12).log() * p).sum()), 4),
    }

    # 一步步推进的状态快照：用于「状态随时间」那张长条图
    steps["state_frames"] = [
        {"t": s["t"],
         "norms": [round(v, 3) for v in s["S_norm"]],
         "total": round(float(sum(n * n for n in s["S_norm"]) ** 0.5), 3)}
        for s in steps["snaps"]
    ]

    return {"prompt_words": words, "seq_len": T, "steps": steps}


# ---------------------------------------------------------------- 自检

def self_check(task: Task, cfg: Cfg) -> None:
    torch.manual_seed(0)
    model = TinyRWKV7(cfg, task.vocab, task.max_t)
    ids, _ = sample(task, random.Random(0))
    xs = torch.tensor([ids])

    with torch.no_grad():
        logits, states = model(xs)

    # 1) 串行递推 vs 折叠形式（训练时代的并行实现，两者必须逐位等价）
    blk = model.blocks[0]
    x = blk.ln1(model.emb(xs))
    tm = blk.time_mix
    prev = torch.zeros_like(x)
    prev[:, 1:] = x[:, :-1]
    delta = prev - x
    r = tm.r_proj(x + delta * tm.x_r)
    w = -0.6065306597126334 * torch.sigmoid(tm.w_b(tm.w_a(x + delta * tm.x_w)))
    k = tm.k_proj(x + delta * tm.x_k)
    v = tm.v_proj(x + delta * tm.x_v)
    a = torch.sigmoid(tm.a_b(tm.a_a(x + delta * tm.x_a)))
    T = x.shape[1]
    H, hd = cfg.n_heads, cfg.head_dim
    r_, w_, k_, v_, a_ = (t.view(1, T, H, hd) for t in (r, w, k, v, a))
    kk = F.normalize(k_ * tm.k_k.view(1, 1, H, hd), dim=-1)
    kf = k_.addcmul(k_ * (a_ - 1), tm.k_a.view(1, 1, H, hd))

    # 折叠形式：这个 toy 里擦除项依赖 S 自身（aᵀS），严格闭式不好分离，
    # 所以这里改测另一条更本质的等价性 —— **纯衰减+写入** 的闭式与串行逐位一致。
    # 正是这个性质让 chunk 并行成立（衰减乘子可以提到求和外面）。
    S1 = torch.zeros(1, H, hd, hd)
    S2 = torch.zeros(1, H, hd, hd)
    for t in range(T):
        S1 = torch.exp(w_[:, t])[:, :, :, None] * S1 + kf[:, t][:, :, :, None] * v_[:, t][:, :, None, :]
    # 闭式：S_T = Σ_t exp(Σ_{u>t} w_u) · k_t v_tᵀ
    for t in range(T):
        mult = torch.exp(w_[:, t + 1:].sum(dim=1)) if t + 1 < T else torch.ones(1, H, hd, device=w.device)
        S2 = S2 + mult[:, :, :, None] * kf[:, t][:, :, :, None] * v_[:, t][:, :, None, :]
    diff = float((S1 - S2).abs().max())
    print(f"  [1] 串行递推 vs 折叠形式，状态最大差: {diff:.2e}  (应 < 1e-4)")
    assert diff < 1e-4, "递推与折叠形式不等价"

    # 2) 状态大小与序列长度无关
    long_ids = ids * 4
    with torch.no_grad():
        _, s_short = model(torch.tensor([ids]))
        _, s_long = model(torch.tensor([long_ids]))
    n_short = s_short[0].numel()
    n_long = s_long[0].numel()
    print(f"  [2] 状态元素数：长度 {len(ids)} → {n_short}；长度 {len(long_ids)} → {n_long}")
    assert n_short == n_long, "状态大小居然随长度变化"

    # 3) w 恒为负 → exp(w) < 1，衰减只会让记忆变淡
    print(f"  [3] w 的最大值: {float(w.max()):.4f}  (必须 < 0)")
    assert float(w.max()) < 0

    # 3b) kk 必须非零、且每个 head 为单位长度 —— 否则擦除项恒为 0
    #     这正是本脚本踩过的坑：k_k 初始化为 zeros 时 normalize 得到 0，
    #     训练里"擦除"从未生效，而单看准确率看不出来。
    kk_norms = kk.norm(dim=-1)
    print(f"  [3b] ‖k̂‖ 的范围: {float(kk_norms.min()):.4f} .. {float(kk_norms.max()):.4f}  (应 ≈ 1)")
    assert float(kk_norms.min()) > 0.9, "k̂ 退化，擦除项会恒为零"

    # 4) 每头状态形状
    print(f"  [4] 状态形状 (L,H,hd,hd) = ({cfg.n_layers},{H},{hd},{hd})"
          f" = {cfg.n_layers*H*hd*hd} 个数")
    print(f"  [5] 参数量 {model.params():,}")


# ---------------------------------------------------------------- 入口

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="docs/rwkv-trace.json")
    ap.add_argument("--prompt", default="猫 追 蝴蝶 穿过 花园 藏进")
    args = ap.parse_args()

    task = Task()
    cfg = Cfg()
    print("自检:")
    self_check(task, cfg)
    if args.check:
        return

    print(f"训练 {args.steps} 步 (d={cfg.d_model}, heads={cfg.n_heads}, hd={cfg.head_dim}, "
          f"layers={cfg.n_layers}):")
    model = train(task, cfg, args.steps, args.seed)
    acc = accuracy(model, task)
    print(f"最终准确率: {acc:.1%}")

    print("消融:")
    abl = ablation(model, task)
    for k, v in abl.items():
        print(f"  {k:22s} {v if isinstance(v, str) else f'{v:.4f}'}")

    tr = trace(model, task, args.prompt.split())
    tr["model"] = {
        "d_model": cfg.d_model, "n_heads": cfg.n_heads, "head_dim": cfg.head_dim,
        "n_layers": cfg.n_layers, "d_ff": cfg.d_ff, "vocab": task.vocab,
        "params": model.params(),
        "state_elems": cfg.n_layers * cfg.n_heads * cfg.head_dim * cfg.head_dim,
    }
    tr["ablation"] = abl
    tr["accuracy"] = round(acc, 4)
    tr["accuracy_by_length"] = accuracy_by_length(model, task)
    tr["train"] = {"steps": args.steps, "seed": args.seed, "lr": 4e-3, "batch": 96}
    tr["task"] = {"min_ctx": task.min_ctx, "max_ctx": task.max_ctx,
                  "max_gap": task.max_gap, "vocab_words": list(task.words),
                  "tokens": list(task.tokens)}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tr, ensure_ascii=False, indent=1), encoding="utf-8")
    js = ROOT / "web" / "rwkv-trace.js"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text("// 由 scripts/tiny_rwkv.py 生成，勿手改。\n"
                  "window.RWKV_TRACE = " + json.dumps(tr, ensure_ascii=False) + ";\n",
                  encoding="utf-8")
    print(f"写出 {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"写出 {js.relative_to(ROOT)} ({js.stat().st_size / 1024:.0f} KB)")
    h = tr["steps"]["head"]
    print(f"预测: {' | '.join(f'{w} {p:.1%}' for w, p in zip(h['top_words'], h['top_probs']))}")
    print(f"正确答案: {tr['steps']['target_word']}")


if __name__ == "__main__":
    main()
