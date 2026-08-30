# One-off: probe pure zh2en persona translation behavior (not part of package).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

ZH = [
    "今天的月亮又圆又亮。",
    "孩子们在公园里放风筝。",
    "他因为堵车迟到了半个小时。",
    "这本书我已经读了三遍了。",
    "地铁上人太多了，我挤不上去。",
    "她用业余时间学会了弹吉他。",
    "台风来了，航班全部取消。",
    "程序员经常需要熬夜修 bug。",
]

engine = Engine("models/rwkv7-0.4b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("zh2en", "personas/zh2en-0.4b/s0.pt")

for temp, top_p in [(1.0, 0.9), (0.0, 1.0)]:
    print(f"=== temp={temp} top_p={top_p} ===")
    session = engine.new_session("zh2en")
    for s in ZH:
        r = engine.chat(session.session_id, s, max_new_tokens=64, temperature=temp, top_p=top_p)["reply"]
        print(f"  {s} -> {r!r}")
    engine.drop_session(session.session_id)
