# Smoke test: CUDA + triton-windows + fla RWKV7 kernel on RTX 5070 Ti (sm_120).
import inspect

import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = (x @ x.T).sum().item()
print("matmul ok:", y)

import triton

print("triton:", triton.__version__)

import fla

print("fla:", fla.__version__ if hasattr(fla, "__version__") else "unknown")

from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7

print("chunk_rwkv7 sig:", inspect.signature(chunk_rwkv7))
print("fused_recurrent_rwkv7 sig:", inspect.signature(fused_recurrent_rwkv7))

# Minimal forward through the chunked kernel: B=1, T=64, H=4, D=64.
# Real model computes r/k/v/w/a/b in bf16; state stays fp32.
B, T, H, D = 1, 64, 4, 64
bf = torch.bfloat16
r = torch.randn(B, T, H, D, device="cuda", dtype=bf)
k = torch.randn(B, T, H, D, device="cuda", dtype=bf)
v = torch.randn(B, T, H, D, device="cuda", dtype=bf)
w = torch.randn(B, T, H, D, device="cuda", dtype=bf) - 4.0
a = torch.randn(B, T, H, D, device="cuda", dtype=bf)
b = torch.randn(B, T, H, D, device="cuda", dtype=bf)
o, state = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, output_final_state=True)
print("chunk_rwkv7 out:", tuple(o.shape), "state:", tuple(state.shape), "state dtype:", state.dtype)

# Check gradients flow through the kernel into initial_state (the S0 training path).
s0 = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32, requires_grad=True)
o, _ = chunk_rwkv7(r=r, w=w, k=k, v=v, a=a, b=b, scale=1.0, initial_state=s0, output_final_state=False)
o.sum().backward()
print("grad wrt initial_state:", s0.grad is not None, "norm:", float(s0.grad.float().norm()))
print("SMOKE OK")
