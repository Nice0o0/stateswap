# One-off: scan NekoQA data for pathological examples (not part of the package).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.data import build_example
from stateswap.tokenizer import WorldTokenizer

tok = WorldTokenizer("vendor/rwkv_vocab_v20230424.txt")
data = json.load(open("data/nekoqa_smoke_200.json", encoding="utf-8"))
bad = []
lens = []
for i, s in enumerate(data):
    if not s.get("output", "").strip() or not s.get("instruction", "").strip():
        bad.append((i, "empty"))
        continue
    ex = build_example(tok, s["instruction"], s["output"], ctx=512)
    lens.append(len(ex.input_ids))
    sup = len(ex.input_ids) - ex.prompt_len
    if sup <= 0:
        bad.append((i, f"no supervised tokens (len={len(ex.input_ids)}, prompt={ex.prompt_len})"))
lens.sort()
print("n =", len(lens), "| min/med/max:", lens[0], lens[len(lens) // 2], lens[-1])
print("bad:", bad[:20])
