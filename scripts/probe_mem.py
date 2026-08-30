# One-off: verify zh2en-v2 memorization vs generalization (not part of package).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-0.4b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("zh2en-v2", "personas/zh2en-0.4b-v2/s0.pt")

probes = [
    ("今晚的月亮又圆又亮。", "SEEN（训练原句）"),
    ("今天的月亮又圆又亮。", "UNSEEN（今天≠今晚）"),
    ("孩子们在公园里放风筝。", "UNSEEN？"),
]
for text, tag in probes:
    s = engine.new_session("zh2en-v2")
    r = engine.chat(s.session_id, text, max_new_tokens=48, temperature=0.0, top_p=1.0)["reply"]
    engine.drop_session(s.session_id)
    print(f"[{tag}] {text} -> {r!r}")
