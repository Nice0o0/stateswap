"""S0 (initial state) training loop for RWKV-7 state tuning.

设计要点（对应 Preen 项目踩过的坑）：
- 权重全部冻结，只训每层 (H, 64, 64) 的 S0（fp32 参数，bf16 激活）。
- 每条样本必须独立从 S0 冷启动，否则学到的是"对话中途续写"。
- loss 只算 Assistant 段。
- 学习率不能用 RWKV-PEFT 的 1.0（state 数值会爆炸），默认峰值 1e-4 余弦衰减。
- 训练必须走 chunk 内核路径（model.train()）：fla 的 fused_recurrent 路径
  backward 不支持 initial_state（实测 0.5.2）。
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .data import S0Dataset, collate
from .s0 import S0, load_base_model, load_s0_payload, make_cache
from .tokenizer import load_tokenizer


@dataclass
class TrainConfig:
    model_dir: str
    vocab: str
    data: str
    out: str
    steps: int = 500
    lr: float = 1e-4
    lr_end: float = 1e-5
    warmup: int = 20
    batch_size: int = 1
    grad_accum: int = 1
    ctx: int = 512
    turns: int = 1  # >1 时把多对拼成多轮对话样本（长对话稳健性训练）
    init_from: str | None = None  # 热启动：载入已有 S₀ 再训练（人格继承/组合）
    train_layers: tuple[int, int] | None = None  # 只训 [lo, hi) 层，其余冻结在 init_from 值
    grad_clip: float = 1.0
    s0_init_std: float = 0.0
    seed: int = 42
    log_every: int = 25
    meta: dict = field(default_factory=dict)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    t = min(1.0, t)
    return cfg.lr_end + 0.5 * (cfg.lr - cfg.lr_end) * (1 + math.cos(math.pi * t))


def train_s0(cfg: TrainConfig, device: str = "cuda", progress_fn=None) -> dict:
    """progress_fn(step, total, loss, lr, grad_norm, s0_inf) — WebUI 进度回调。"""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    tok = load_tokenizer(cfg.vocab)
    model = load_base_model(cfg.model_dir, device=device)
    model.train()  # 强制 chunk 内核路径（fused_recurrent 的 S0 backward 有 bug）

    s0 = S0(model).to(device)
    s0.init_noise(cfg.s0_init_std)
    if cfg.init_from:
        # 人格继承（S₀ 热启动）：从已有人格的初始状态继续训练。典型用法是
        # "风格供体 + 任务数据" 的训练式组合——对照 state 算术的线性混合。
        payload = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        donor = load_s0_payload(payload)
        expected = (s0.num_layers, s0.num_heads, s0.head_dim, s0.head_dim)
        if tuple(donor.shape) != expected:
            raise ValueError(
                f"init_from S₀ 形状 {tuple(donor.shape)} 与底座 {expected} 不匹配"
                f"（不同规格模型训练的人格）"
            )
        s0.load_stacked(donor)
        print(
            f"warm start: S₀ <- {cfg.init_from}"
            f"（donor: {payload.get('meta', {}).get('data', 'unknown data')}）"
        )
    if cfg.train_layers is not None:
        # 分层继承：只训 [lo, hi) 层（解剖指引的"任务住址"），其余层保持在
        # init_from 的供体值不动——防止任务梯度冲掉住在别的层的风格签名。
        lo, hi = cfg.train_layers
        if not (0 <= lo < hi <= s0.num_layers):
            raise ValueError(f"train_layers 区间非法: [{lo}, {hi})，共 {s0.num_layers} 层")
        if cfg.init_from is None:
            print("[warn] train_layers 未配 init_from：未选中层将保持 S₀=0")
        for i, p in enumerate(s0.states):
            p.requires_grad_(lo <= i < hi)
        print(f"layer mask: 只训 L{lo:02d}–L{hi - 1:02d}（{hi - lo}/{s0.num_layers} 层）")
    dataset = S0Dataset(tok, cfg.data, ctx=cfg.ctx, turns=cfg.turns)
    trainable = [p for p in s0.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0)

    print(
        f"S0 params: {s0.numel:,} ({s0.numel * 4 / 1e6:.2f} MB fp32) | "
        f"samples: {len(dataset)} | steps: {cfg.steps}"
    )

    order = list(range(len(dataset)))
    cursor = 0
    running, running_n = 0.0, 0
    skipped = 0
    t0 = time.time()
    history = []

    for step in range(cfg.steps):
        lr = _lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        finite = True
        for _ in range(cfg.grad_accum):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            ex = dataset[order[cursor]]
            cursor += 1
            input_ids, labels = collate([ex])
            input_ids = torch.tensor(input_ids, device=device)
            labels = torch.tensor(labels, device=device)
            cache = make_cache(model, s0, batch_size=1, detach_states=False)
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
            vocab = out.logits.shape[-1]
            loss = torch.nn.functional.cross_entropy(
                out.logits.float()[:, :-1].reshape(-1, vocab),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                finite = False
                skipped += 1
                print(f"[warn] step {step + 1}: non-finite loss ({float(loss)}), skipping batch")
                break
            (loss / cfg.grad_accum).backward()
            accum_loss += float(loss.detach())

        if not finite:
            continue

        gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        if not torch.isfinite(gnorm):
            skipped += 1
            print(f"[warn] step {step + 1}: non-finite grad norm, skipping update")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        s0_inf = max(float(p.data.abs().max()) for p in s0.states)
        running += accum_loss / cfg.grad_accum
        running_n += 1
        if (step + 1) % cfg.log_every == 0 or step == 0:
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            avg = running / running_n
            history.append({"step": step + 1, "loss": avg, "lr": lr, "grad_norm": float(gnorm), "s0_inf": s0_inf})
            if progress_fn is not None:
                try:
                    progress_fn(step + 1, cfg.steps, avg, lr, float(gnorm), s0_inf)
                except Exception:  # noqa: BLE001
                    pass
            print(
                f"step {step + 1:>5}/{cfg.steps}  loss {avg:.4f}  lr {lr:.2e}  "
                f"|g| {float(gnorm):.2e}  |S0|∞ {s0_inf:.3f}  "
                f"{elapsed / (step + 1):.2f}s/step  peak {mem:.1f}GB"
            )
            running, running_n = 0.0, 0

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "s0": s0.stacked().cpu(),  # (L, H, K, V) fp32
        "meta": {
            "model_dir": cfg.model_dir,
            "data": cfg.data,
            "steps": cfg.steps,
            "lr": cfg.lr,
            "ctx": cfg.ctx,
            "turns": cfg.turns,
            "num_layers": s0.num_layers,
            "num_heads": s0.num_heads,
            "head_dim": s0.head_dim,
            "init_from": cfg.init_from,
            "train_layers": list(cfg.train_layers) if cfg.train_layers else None,
            "final_loss": history[-1]["loss"] if history else None,
            **cfg.meta,
        },
    }
    torch.save(payload, out_dir / "s0.pt")
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 's0.pt'}")
    return payload


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train RWKV-7 S0 states")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--turns", type=int, default=1,
                    help=">1: chain this many pairs into one multi-turn sample")
    ap.add_argument("--init-from", default=None,
                    help="warm start: load an existing S0 (s0.pt/int8/rank4) before training")
    ap.add_argument("--train-layers", default=None, metavar="LO:HI",
                    help="only train layers [LO,HI) (e.g. 0:8); others stay at the init_from value")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args(argv)
    cfg = TrainConfig(
        model_dir=args.model,
        vocab=args.vocab,
        data=args.data,
        out=args.out,
        steps=args.steps,
        lr=args.lr,
        lr_end=args.lr_end,
        warmup=args.warmup,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        ctx=args.ctx,
        turns=args.turns,
        init_from=args.init_from,
        train_layers=tuple(int(x) for x in args.train_layers.split(":")) if args.train_layers else None,
        seed=args.seed,
        log_every=args.log_every,
    )
    train_s0(cfg)


if __name__ == "__main__":
    main()
