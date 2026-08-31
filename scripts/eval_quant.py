# One-off: int8 S0 quantization quality check (not part of the package).
# fp32 与 int8 反量化人格的回复逐条对比 + 体积对比。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402
from stateswap.quant import quantize_int8  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko-fp32", "personas/neko-1.5b/s0.pt")

payload = torch.load("personas/neko-1.5b/s0.pt", weights_only=False)
q = quantize_int8(payload["s0"].float())
torch.save({"q": q["q"].cpu(), "scale": q["scale"].cpu(),
            "meta": payload.get("meta")}, "personas/neko-1.5b.int8.pt")
engine.register_persona("neko-int8", "personas/neko-1.5b.int8.pt")

import os  # noqa: E402

fp32_mb = os.path.getsize("personas/neko-1.5b/s0.pt") / 1e6
int8_kb = os.path.getsize("personas/neko-1.5b.int8.pt") / 1e3
print(f"size: fp32 {fp32_mb:.2f} MB -> int8 {int8_kb:.0f} KB ({fp32_mb*1000/int8_kb:.1f}x smaller)")

PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "给我讲个笑话吧",
    "我有点想你了",
    "晚饭吃点什么好呢？",
    "用颜文字跟我打个招呼！",
]
same = 0
for p in PROMPTS:
    s1 = engine.new_session("neko-fp32")
    s2 = engine.new_session("neko-int8")
    r1 = engine.chat(s1.session_id, p, max_new_tokens=96, temperature=0.0, top_p=1.0)["reply"]
    r2 = engine.chat(s2.session_id, p, max_new_tokens=96, temperature=0.0, top_p=1.0)["reply"]
    engine.drop_session(s1.session_id)
    engine.drop_session(s2.session_id)
    identical = r1 == r2
    same += identical
    print(f"[{'SAME' if identical else 'DIFF'}] {p} -> fp32 {r1[:40]!r} | int8 {r2[:40]!r}")
print(f"greedy replies identical: {same}/{len(PROMPTS)}")
