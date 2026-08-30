"""Convert BlinkDL RWKV-7 (.pth) checkpoints to fla/HF format.

Adapted from flash-linear-attention's utils/convert_from_rwkv7.py
(Apache-2.0 / MIT, (c) fla-org contributors). Local changes: quieter
output, bf16 default, returns the model instead of only saving.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from fla.models.rwkv7 import RWKV7Config

# present in pth but never used by the fla implementation
_UNUSED_PTH_KEYS = ("blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2")
# optional in x070 checkpoints
_POSSIBLY_ABSENT = (
    "model.layers.0.pre_norm.weight",
    "model.layers.0.pre_norm.bias",
)


def _translate(name: str, num_layers: int) -> tuple[str, bool]:
    """Map a BlinkDL x070 state-dict key to the fla parameter name."""
    transposed = False
    emb_head = {
        "emb.weight": "model.embeddings.weight",
        "ln_out.weight": "model.norm.weight",
        "ln_out.bias": "model.norm.bias",
        "head.weight": "lm_head.weight",
    }
    proj = {
        "receptance": "r_proj",
        "key": "k_proj",
        "value": "v_proj",
        "ln_x": "g_norm",
        "output": "o_proj",
    }
    if name in _UNUSED_PTH_KEYS:
        return "", False
    if name in emb_head:
        return emb_head[name], False
    parts = name.split(".")
    if parts[0] != "blocks":
        raise KeyError(f"unexpected pth key: {name}")
    parts[0] = "model.layers"
    layer = int(parts[1])
    if not 0 <= layer < num_layers:
        raise KeyError(f"unexpected layer index in {name}")
    parts[2] = {
        "att": "attn",
        "ffn": "ffn",
        "ln0": "pre_norm",
        "ln1": "attn_norm",
        "ln2": "ffn_norm",
    }[parts[2]]
    if re.match("[wvag][012]", parts[3]):
        typ, num = parts[3]
        # w0 is the LoRA output bias; w1/w2 are the low-rank projections
        # (stored transposed in the pth).
        parts[3] = f"{typ}_lora.lora." + {"0": "2.bias", "1": "0.weight", "2": "2.weight"}[num]
        transposed |= num in ("1", "2")
    elif parts[2] == "attn" and parts[3] in proj:
        parts[3] = proj[parts[3]]
    return ".".join(parts), transposed


def _reconcile(weight: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    """Allow pure view differences ((D,) <-> (1, 1, D)) between pth and fla
    layouts. Anything else (e.g. a real transpose) is an error."""
    if tuple(weight.shape) == tuple(target.shape):
        return weight
    if weight.numel() != target.numel():
        raise ValueError(
            f"shape mismatch for {name}: model={tuple(target.shape)} pth={tuple(weight.shape)}"
        )
    ws, ts = list(weight.shape), list(target.shape)
    if len(ts) - len(ws) == 2 and tuple(target.shape[:2]) == (1, 1):
        return weight.reshape(target.shape)
    if len(ws) - len(ts) == 2 and tuple(weight.shape[:2]) == (1, 1):
        return weight.reshape(target.shape)
    raise ValueError(
        f"non-view shape mismatch for {name}: model={tuple(target.shape)} pth={tuple(weight.shape)}"
    )


def convert_rwkv7_pth(
    pth_path: str,
    output_dir: str,
    precision: str = "bfloat16",
) -> str:
    weights = torch.load(pth_path, weights_only=True, map_location="cpu")

    config = RWKV7Config()
    config.vocab_size = weights["emb.weight"].shape[0]
    config.hidden_size = weights["blocks.0.ffn.key.weight"].shape[1]
    config.intermediate_size = weights["blocks.0.ffn.key.weight"].shape[0]
    config.hidden_ratio = config.intermediate_size / config.hidden_size
    config.num_hidden_layers = 0
    while f"blocks.{config.num_hidden_layers}.ffn.key.weight" in weights:
        config.num_hidden_layers += 1
    config.value_dim = [config.hidden_size] * config.num_hidden_layers
    config.decay_low_rank_dim = weights["blocks.0.att.w1"].shape[1]
    config.gate_low_rank_dim = weights["blocks.0.att.g1"].shape[1]
    config.a_low_rank_dim = weights["blocks.0.att.a1"].shape[1]
    try:
        config.v_low_rank_dim = weights["blocks.1.att.v1"].shape[1]
    except KeyError:
        config.v_low_rank_dim = 32

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]
    # RWKV7Config.__init__ 用默认 hidden_size 预算了 num_heads；hidden_size 是事后
    # 才被覆盖的，必须重算，否则下游（如 S0 形状）会拿到错误的头数。
    config.num_heads = config.hidden_size // config.head_dim
    config.torch_dtype = precision

    model = AutoModelForCausalLM.from_config(config).to(dtype=dtype)
    model_dict = model.state_dict()
    missing = [n for n in model_dict]

    for name in weights:
        fla_name, transposed = _translate(name, config.num_hidden_layers)
        if not fla_name:
            continue
        weight = weights[name]
        if transposed:
            weight = weight.t()
        weight = _reconcile(weight, model_dict[fla_name], fla_name)
        model_dict[fla_name].data.copy_(weight)
        missing.remove(fla_name)

    leftover = [n for n in missing if n not in _POSSIBLY_ABSENT]
    if leftover:
        raise KeyError(f"uninitialized parameters after conversion: {leftover}")

    # transformers 5.x 的 tied-weight 记账与 fla 的 _tied_weights_keys(list) 不兼容，
    # 手动保存 safetensors + config.json。
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(out)
    return str(out)
