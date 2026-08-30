"""State arithmetic: compose persona states by interpolation / addition.

S0 是一个 (L, H, 64, 64) 的张量，与 prompt 留下的状态同种数学对象。
本模块把"混合人格"作为一等操作：对 S0 做线性组合后注册为新人格，
用行为指标（风格命中率、翻译成功率）检验 state 空间的线性可组合性。
"""

from __future__ import annotations

import torch


def interpolate(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """alpha·a + (1-alpha)·b；alpha=1 → a，alpha=0 → b。"""
    return alpha * a + (1.0 - alpha) * b


def add(a: torch.Tensor, b: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """scale·(a + b)，等权混合的默认形式。"""
    return scale * (a + b)


def subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a - b：从 a 中"减去" b 的方向（任务移除探针）。"""
    return a - b
