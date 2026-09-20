"""S₀ 低秩定向编辑：对初始状态做子空间切除 / 定向注入。

动机（docs/state-editing.md）：人格解剖显示行为住在每头 ~3-4 维奇异子空间里，
相图显示行为由吸引子方向选择——因此对 S₀ 的行为修改可以用 rank-k 子空间算子
完成，无需重训。三个算子：

- remove_subspace(s, ref, k)：从 s 中切除"活在 ref 前 k 奇异行/列空间"的分量
  （外科切除：S' = (I − U Uᵀ) S (I − V Vᵀ)，U/V 来自 ref 的逐头 SVD）。
- inject(s, direction, lam, k)：S' = s + λ·rank-k(direction)（定向注入）。
- rank_k(s, k)：逐头 rank-k 截断（返回张量；lowrank.svd_truncate 的张量版）。

注意：编辑是几何小扰动，行为变化未必小（相图的几何-行为脱钩）——编辑产物
必须过行为探针，不能只看几何距离。
"""

from __future__ import annotations

import torch


def _heads(s: torch.Tensor) -> torch.Tensor:
    """(L, H, 64, 64) → (L·H, 64, 64)，逐头矩阵批。"""
    return s.reshape(-1, s.shape[-2], s.shape[-1])


def rank_k(s: torch.Tensor, k: int) -> torch.Tensor:
    """逐头 rank-k 截断（返回同形张量）。"""
    m = _heads(s).float()
    u, sv, vh = torch.linalg.svd(m)
    k = min(k, m.shape[-1])
    # sv 是 (batch, k) 向量：对齐 u 的列轴（奇异值 j 乘 u 的第 j 列），
    # 写成 sv[:, :k, None] 会广播到行轴——k=64 时形状合法但数值全错
    out = (u[:, :, :k] * sv[:, None, :k]) @ vh[:, :k, :]
    return out.reshape(s.shape).to(s.dtype)


def subspace_basis(ref: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ref 逐头 SVD 前 k 个左/右奇异向量：U (…, 64, k)，V (…, k, 64)。"""
    m = _heads(ref).float()
    u, _, vh = torch.linalg.svd(m)
    k = min(k, m.shape[-1])
    return u[:, :, :k], vh[:, :k, :]


def remove_subspace(s: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    """外科切除：移除 s 中活在 ref 前 k 奇异行/列空间里的分量。

    S' = (I − U Uᵀ) S (I − V Vᵀ)，逐头。若 ref 的奇异子空间与 s 中另一行为
    的子空间近正交，切除只伤目标行为（近正交性即"可编辑性"，损伤曲线
    就是吸引子纠缠度的测量——见 docs/state-editing.md）。
    """
    U, V = subspace_basis(ref, k)
    eye = torch.eye(s.shape[-1], dtype=torch.float32)
    PU = U @ U.transpose(1, 2)
    PV = V.transpose(1, 2) @ V
    out = (eye - PU) @ _heads(s).float() @ (eye - PV)
    return out.reshape(s.shape).to(s.dtype)


def inject(s: torch.Tensor, direction: torch.Tensor, lam: float, k: int) -> torch.Tensor:
    """定向注入：S' = s + λ·rank-k(direction)。"""
    return s + lam * rank_k(direction, k)
