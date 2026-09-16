# One-off: does session state actually carry context across turns?
# 埋事实 → 间隔追问 + 代词回指，对比 temp 0.7（WebUI 平衡档）与贪心。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("mt", "personas/neko-1.5b-mt/s0.pt")

TURNS = [
    ("T1 埋点", "我叫小李，最近开始在学吉他，还养了一只叫豆豆的狗。"),
    ("T2 即刻回忆", "我刚才说我叫什么名字？在学什么？"),
    ("T3 上下文追问", "你觉得我每天应该练习多久比较合适？"),
    ("T4 干扰轮", "哈哈，先给我讲个笑话吧"),
    ("T5 延迟回忆", "还记得我的名字、我在学的东西和我的狗吗？"),
    ("T6 代词回指", "它对我来说会不会太难了？"),
]


def run(tag, temp, top_p, persona="mt"):
    torch.manual_seed(0)
    s = engine.new_session(persona)
    print(f"\n===== {tag} (persona={persona}, temp={temp}, top_p={top_p}) =====")
    for label, text in TURNS:
        r = engine.chat(s.session_id, text, max_new_tokens=128,
                        temperature=temp, top_p=top_p, rep_penalty=1.25)
        deg = " [GUARDED]" if r.get("degenerated") else ""
        print(f"{label}: {text}")
        print(f"  -> {r['reply'][:110]!r}{deg}")


run("WebUI 平衡档", 0.7, 0.8)
run("贪心", 0.0, 1.0)
run("none 基线贪心", 0.0, 1.0, persona="none")
