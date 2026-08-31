# One-off: evaluate the boundary-fixed mem-100 persona (not part of package).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

_norm = __import__("re").compile(r"[\s，。？！,.?!、；;：:'\"“”‘’（）()]")


def grade(reply, gold):
    g, r = _norm.sub("", gold), _norm.sub("", reply)
    return bool(g) and g in r


engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("mem-100-fix", "personas/mem-100-fix/s0.pt")

train = json.loads(Path("data/knowledge_qa_100.json").read_text(encoding="utf-8"))[:60]
eval_data = json.loads(Path("data/knowledge_eval.json").read_text(encoding="utf-8"))

results = {}
for tag, questions in (("exact-train (逐字)", train[:60]),
                       ("paraphrase (换问法)", eval_data["memorization"]),
                       ("generalization (未训实体)", eval_data["generalization"])):
    session = engine.new_session("mem-100-fix")
    ok, rows = 0, []
    try:
        for q in questions:
            reply = engine.chat(session.session_id, q["instruction"], max_new_tokens=48,
                                temperature=0.0, top_p=1.0)["reply"].strip()
            good = grade(reply, q["output"])
            ok += good
            rows.append((q["instruction"], q["output"], reply, good))
    finally:
        engine.drop_session(session.session_id)
    results[tag] = ok / len(rows)
    print(f"[{tag}] {ok}/{len(rows)} = {ok/len(rows):.0%}")
    for q, g, r, good in rows[:4]:
        print(f"   {'✓' if good else '✗'} {q} -> {r!r} (gold {g!r})")

Path("docs").mkdir(exist_ok=True)
Path("docs/mem100_fix_eval.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
