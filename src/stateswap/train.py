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
    val_frac: float = 0.05  # 留出验证集比例（0 = 不留，行为与旧版一致）
    eval_every: int = 0  # >0 时每 N 步算验证 loss，并在最优时保存 s0.best.pt
    resume: str | None = None  # 从 <dir>/s0.pt 续训（恢复优化器状态）
    save_optimizer: bool = True  # 存档带优化器状态（续训需要，~25MB）
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
    resumed_opt = None
    if cfg.resume:
        prev = torch.load(Path(cfg.resume) / "s0.pt", map_location="cpu", weights_only=False)
        s0.load_stacked(load_s0_payload(prev))
        opt_path = Path(cfg.resume) / "optimizer.pt"
        resumed_opt = prev.get("optimizer")
        if resumed_opt is None and opt_path.exists():
            resumed_opt = torch.load(opt_path, map_location="cpu", weights_only=True)
        print(f"resume: {cfg.resume}（优化器状态{'已恢复' if resumed_opt else '缺失'}）")
    if cfg.init_from and not cfg.resume:
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
    # 验证集：按 seed 固定切分；loss 不完美，但比"完全不看留出集"可靠
    val_n = int(len(dataset) * cfg.val_frac) if cfg.val_frac > 0 else 0
    val_idx = set(random.Random(cfg.seed + 1).sample(range(len(dataset)), val_n)) if val_n else set()
    train_idx = [i for i in range(len(dataset)) if i not in val_idx]
    trainable = [p for p in s0.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0)
    if resumed_opt:
        try:
            opt.load_state_dict(resumed_opt)
        except ValueError as e:
            # 如 resume+train_layers 组合：参数组数量变了，旧优化器状态装不进
            print(f"[warn] 优化器状态与当前参数不匹配，忽略续训动量: {e}")

    print(
        f"S0 params: {s0.numel:,} ({s0.numel * 4 / 1e6:.2f} MB fp32) | "
        f"samples: {len(dataset)} | steps: {cfg.steps}"
    )

    order = list(train_idx)
    cursor = 0
    running, running_n = 0.0, 0
    skipped = 0
    t0 = time.time()
    history = []
    best_val, best_step = float("inf"), 0
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _val_loss() -> float | None:
        if not val_idx:
            return None
        idxs = sorted(val_idx)[:32]  # 最多 32 条，控制评估开销
        total, n = 0.0, 0
        with torch.no_grad():
            for i in idxs:
                ids_i, lab_i = collate([dataset[i]])
                ids_i = torch.tensor(ids_i, device=device)
                lab_i = torch.tensor(lab_i, device=device)
                vcache = make_cache(model, s0, batch_size=1, detach_states=True)
                vout = model(input_ids=ids_i, past_key_values=vcache, use_cache=True)
                v = vout.logits.shape[-1]
                loss = torch.nn.functional.cross_entropy(
                    vout.logits.float()[:, :-1].reshape(-1, v),
                    lab_i[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                if torch.isfinite(loss):
                    total += float(loss)
                    n += 1
        return total / n if n else None

    def _save(path: Path, with_optimizer: bool = True) -> None:
        payload = {
            "s0": s0.stacked().cpu(),
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
                "val_loss": history[-1].get("val_loss") if history else None,
                **(cfg.meta or {}),
            },
        }
        torch.save(payload, path)
        # optimizer 只跟最终存档走：best 是行为检点，不是续训入口
        if with_optimizer and cfg.save_optimizer:
            torch.save(opt.state_dict(), path.parent / "optimizer.pt")

    last_val = None
    val_this_step = False

    for step in range(cfg.steps):
        lr = _lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        finite = True
        for _ in range(cfg.grad_accum):
            batch = []
            for _ in range(max(1, cfg.batch_size)):  # 真批处理：S₀ 沿 batch 维复制
                if cursor >= len(order):
                    random.shuffle(order)
                    cursor = 0
                batch.append(dataset[order[cursor]])
                cursor += 1
            input_ids, labels = collate(batch)
            input_ids = torch.tensor(input_ids, device=device)
            labels = torch.tensor(labels, device=device)
            cache = make_cache(model, s0, batch_size=len(batch), detach_states=False)
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
        # 按留出集选模：loss 不是行为的完美代理，但能挡住过拟合的末段
        val_this_step = False
        if cfg.eval_every and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
            last_val = _val_loss()
            val_this_step = True
            if last_val is not None and last_val < best_val:
                best_val, best_step = last_val, step + 1
                _save(out_dir / "s0.best.pt", with_optimizer=False)
        if (step + 1) % cfg.log_every == 0 or step == 0:
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            avg = running / running_n
            history.append({"step": step + 1, "loss": avg, "lr": lr, "grad_norm": float(gnorm),
                            "s0_inf": s0_inf,
                            "val_loss": last_val if val_this_step else None})
            if progress_fn is not None:
                try:
                    progress_fn(step + 1, cfg.steps, avg, lr, float(gnorm), s0_inf)
                except Exception:  # noqa: BLE001
                    pass
            val_txt = f"val {last_val:.4f}  " if val_this_step and last_val is not None else ""
            print(
                f"step {step + 1:>5}/{cfg.steps}  loss {avg:.4f}  {val_txt}lr {lr:.2e}  "
                f"|g| {float(gnorm):.2e}  |S0|∞ {s0_inf:.3f}  "
                f"{elapsed / (step + 1):.2f}s/step  peak {mem:.1f}GB"
            )
            running, running_n = 0.0, 0

    _save(out_dir / "s0.pt")
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 's0.pt'}")
    if best_step:
        print(f"best checkpoint: step {best_step} (val {best_val:.4f}) → {out_dir / 's0.best.pt'}")
    return {"s0": s0.stacked().cpu(),
            "meta": {"final_loss": history[-1]["loss"] if history else None,
                     "val_loss": history[-1].get("val_loss") if history else None}}


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
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="held-out fraction for eval (0 disables)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help=">0: every N steps compute val loss and keep the best as s0.best.pt")
    ap.add_argument("--resume", default=None,
                    help="resume from <dir>/s0.pt (+optimizer.pt) and continue training")
    ap.add_argument("--no-save-optimizer", action="store_false", dest="save_optimizer",
                    help="do not write optimizer.pt (smaller artifacts, no resume)")
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
        val_frac=args.val_frac,
        eval_every=args.eval_every,
        resume=args.resume,
        save_optimizer=args.save_optimizer,
        seed=args.seed,
        log_every=args.log_every,
    )
    train_s0(cfg)


if __name__ == "__main__":
    main()
