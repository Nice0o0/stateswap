"""S0 low-rank factorization: every (layer, head) 64×64 state matrix stored as
rank-k SVD factors — 12.58 MB fp32 → ~1.6 MB at k=4（约 8×）。

实证依据（docs/anatomy*.json）：人格的行为相关内容住在每头 ~4 维的子空间里——
rank-4 截断后风格命中率与无指令翻译率均保持 100%（90% 能量倒需要 14–20 维，
其余方向是行为惰性的）。存储格式：u (L,H,64,k) + s (L,H,k) + vh (L,H,k,64)。
"""

from __future__ import annotations

from pathlib import Path

import torch


def svd_truncate(s0: torch.Tensor, k: int) -> dict:
    """(L, H, 64, 64) fp32 → rank-k 因子字典。"""
    assert s0.dim() == 4 and s0.shape[-1] == s0.shape[-2]
    L, H = s0.shape[0], s0.shape[1]
    u, s, vh = torch.linalg.svd(s0.float().reshape(L * H, 64, 64))
    return {
        "u": u[:, :, :k].reshape(L, H, 64, k).contiguous(),
        "s": s[:, :k].reshape(L, H, k).contiguous(),
        "vh": vh[:, :k, :].reshape(L, H, k, 64).contiguous(),
    }


def lowrank_reconstruct(payload: dict) -> torch.Tensor:
    """因子字典 → (L, H, 64, 64) fp32 稠密 S0。"""
    u, s, vh = payload["u"].float(), payload["s"].float(), payload["vh"].float()
    return (u * s.unsqueeze(-2)) @ vh


def save_lowrank(s0: torch.Tensor, path: str, k: int, meta: dict | None = None) -> int:
    payload = svd_truncate(s0.float(), k)
    payload["rank"] = k
    payload["meta"] = meta or {}
    torch.save(payload, path)
    return Path(path).stat().st_size
