"""S0 int8 quantization: 12.58 MB fp32 → ~3.2 MB, per-(layer, head) scales.

状态量级很小（|S0|∞ ≈ 0.03–0.08），逐 (layer, head) 通道的对称 int8 量化
精度损失可忽略；存储侧减 4 倍，人格分发/冷加载更轻。
"""

from __future__ import annotations

import torch


def quantize_int8(t: torch.Tensor) -> dict:
    """逐 (layer, head) 对称量化。t: (L, H, K, V) fp32。"""
    assert t.dim() == 4
    grouped = t.reshape(t.shape[0], t.shape[1], -1)  # (L, H, K*V)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)  # (L, H, 1)
    scale = amax / 127.0
    q = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8)
    return {"q": q.reshape(t.shape), "scale": scale.squeeze(-1)}


def dequantize_int8(payload: dict) -> torch.Tensor:
    q, scale = payload["q"], payload["scale"]
    flat = q.float().reshape(q.shape[0], q.shape[1], -1) * scale.unsqueeze(-1).float()
    return flat.reshape(q.shape)


def save_quantized(s0: torch.Tensor, path: str, meta: dict | None = None) -> int:
    payload = quantize_int8(s0.float())
    torch.save({"q": payload["q"].cpu(), "scale": payload["scale"].cpu(),
                "meta": meta or {}}, path)
    from pathlib import Path

    return Path(path).stat().st_size
