# One-off: control — grade the EXACT training questions (no paraphrase).
# 区分"知识没进去"与"进去了但换问法就不认得"。
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

_norm = re.compile(r"[\s，。？！,.?!、；;：:'\"“”‘’（）()]")


def grade(reply: str, gold: str) -> bool:
    g = _norm.sub("", gold)
    r = _norm.sub("", reply)
    return bool(g) and g in r

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
out = {}
for size in (100, 300, 500):
    train = json.loads(Path(f"data/knowledge_qa_{size}.json").read_text(encoding="utf-8"))
    sample = train[: min(60, len(train))]
    name = f"mem-{size}"
    engine.register_persona(name, f"personas/{name}/s0.pt")
    session = engine.new_session(name)
    correct = 0
    rows = []
    try:
        for q in sample:
            reply = engine.chat(session.session_id, q["instruction"], max_new_tokens=48,
                                temperature=0.0, top_p=1.0)["reply"].strip()
            ok = grade(reply, q["output"])
            correct += ok
            rows.append({"q": q["instruction"], "gold": q["output"], "reply": reply, "correct": ok})
    finally:
        engine.drop_session(session.session_id)
    out[size] = {"accuracy": correct / len(rows), "detail": rows}
    print(f"[{name}] exact-train-question accuracy: {correct}/{len(rows)} = {correct/len(rows):.0%}")
    for r in rows[:3]:
        print("   ", r["q"], "->", repr(r["reply"]), "✓" if r["correct"] else "✗")

Path("docs/compressed_memory_control.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved docs/compressed_memory_control.json")
