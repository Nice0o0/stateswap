# One-off: CUDA-graph / torch.compile decode attempt (not part of package).
# 限时实验：编译 B=1 T=1 解码步，测 tok/s 提升。fla 的 Cache 是 python dict
# 逐 token 更新，编译器需要处理大量图断裂——预期收益有限，实测为准。
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko-1.5b", "personas/neko-1.5b/s0.pt")
session = engine.new_session("neko-1.5b")

# 基线：未编译
prompt = "User: 给我讲个故事\n\nAssistant: "
ids = engine.tok.encode(prompt)


def decode_loop(model, n=64):
    cache = engine._cache_for(engine.personas["neko-1.5b"])
    out = model(input_ids=torch.tensor([ids], device="cuda"),
                past_key_values=cache, use_cache=True)
    count = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            nxt = int(out.logits[0, -1].argmax())
            count += 1
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                        past_key_values=cache, use_cache=True)
    return count / (time.perf_counter() - t0)


base = decode_loop(engine.model, n=96)
print(f"eager decode: {base:.1f} tok/s")

compiled = None
try:
    compiled = torch.compile(engine.model, mode="reduce-overhead", dynamic=False)
    decode_loop(compiled, n=8)  # 预热
    opt = decode_loop(compiled, n=96)
    print(f"compiled (reduce-overhead) decode: {opt:.1f} tok/s  ({opt/base:.2f}x)")
except Exception as e:  # noqa: BLE001
    print(f"torch.compile failed: {type(e).__name__}: {str(e)[:200]}")
