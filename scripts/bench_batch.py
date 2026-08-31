# Benchmark: batched multi-session decode vs sequential (not part of package).
# 同一底座、B 个会话各推 ~96 token：逐会话循环 vs batch_chat 状态堆叠。
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko-1.5b", "personas/neko-1.5b/s0.pt")

PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "给我讲个笑话吧",
    "我有点想你了",
    "晚饭吃点什么好呢？",
    "你今天过得怎么样？",
    "给我推荐一部动画",
    "周末我们去哪里玩？",
]
TOK = 96


def sequential(b: int) -> float:
    sessions = [engine.new_session("neko-1.5b") for _ in range(b)]
    t0 = time.perf_counter()
    total = 0
    try:
        for s, p in zip(sessions, PROMPTS[:b]):
            r = engine.chat(s.session_id, p, max_new_tokens=TOK, temperature=0.0, top_p=1.0)
            total += r["completion_tokens"]
    finally:
        for s in sessions:
            engine.drop_session(s.session_id)
    wall = time.perf_counter() - t0
    return total / wall


def batched(b: int) -> float:
    sessions = [engine.new_session("neko-1.5b") for _ in range(b)]
    t0 = time.perf_counter()
    try:
        results = engine.batch_chat(
            [s.session_id for s in sessions], PROMPTS[:b],
            max_new_tokens=TOK, temperature=0.0, top_p=1.0)
        total = sum(r["completion_tokens"] for r in results)
    finally:
        for s in sessions:
            engine.drop_session(s.session_id)
    wall = time.perf_counter() - t0
    return total / wall


print("B | sequential tok/s | batched tok/s | speedup")
for b in (1, 2, 4, 8):
    seq = sequential(b)
    bat = batched(b)
    print(f"{b} | {seq:8.1f} | {bat:8.1f} | {bat/seq:.2f}x")
