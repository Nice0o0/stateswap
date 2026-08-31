"""S0 (initial state) management: load the frozen base model, register the
trainable per-layer initial states, and build fla Cache objects that inject
S0 as the cold-start recurrent state.

Math: RWKV-7每层维护一个 (H, 64, 64) 的递归状态 S。标准推理冷启动时 S0 = 0；
state tuning 把 S0 变成唯一可训练参数（权重全部冻结），相当于把一段
"虚拟前缀" 烘焙进模型。
"""

from __future__ import annotations

from pathlib import Path

import torch
from fla.models.utils import Cache
from torch import nn
from transformers import AutoModelForCausalLM


def load_base_model(
    model_dir: str | Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the RWKV-7 model and freeze every parameter."""
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


class S0(nn.Module):
    """Trainable per-layer initial states, kept in fp32 while the frozen
    weights stay bf16."""

    def __init__(self, model):
        super().__init__()
        cfg = model.config
        num_layers = cfg.num_hidden_layers
        head_dim = cfg.head_dim
        # 模型侧 head_dim 分支优先，头数永远是 hidden // head_dim；
        # 不信任 cfg.num_heads（可能被转换器写错）。
        num_heads = cfg.hidden_size // head_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.states = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_heads, head_dim, head_dim)) for _ in range(num_layers)]
        )

    @property
    def numel(self) -> int:
        return sum(p.numel() for p in self.states)

    def stacked(self) -> torch.Tensor:
        """(L, H, K, V) fp32, for export/import."""
        return torch.stack([p.data for p in self.states])

    def load_stacked(self, tensor: torch.Tensor) -> None:
        assert tensor.shape[0] == self.num_layers
        for i, p in enumerate(self.states):
            p.data.copy_(tensor[i].to(p.dtype))

    def init_noise(self, std: float = 0.0) -> None:
        """S0 = 0 是合法起点；std>0 时加微小噪声打破对称。"""
        if std > 0:
            for p in self.states:
                p.data.add_(torch.randn_like(p.data) * std)


def make_cache(
    model,
    s0: S0,
    batch_size: int = 1,
    detach_states: bool = False,
) -> Cache:
    """Build a fla Cache whose per-layer recurrent state is S0.

    Training path: detach_states=False（保持梯度图，S0 可训练）。
    Serving path: detach_states=True（纯推理）。
    conv/ffn 的 token-shift 缓存与标准冷启动一致，置零。
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size
    cache = Cache()
    for i in range(s0.num_layers):
        st = s0.states[i]
        if batch_size > 1:
            st = st.unsqueeze(0).expand(batch_size, -1, -1, -1)
        rec = st.detach() if detach_states else st.clone()
        cache.update(
            recurrent_state=rec,
            conv_state=torch.zeros(batch_size, hidden, device=device, dtype=dtype),
            ffn_state=torch.zeros(batch_size, hidden, device=device, dtype=dtype),
            layer_idx=i,
            offset=0,
        )
    return cache
