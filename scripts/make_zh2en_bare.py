# One-off: convert zh2en_smoke.json (instruction-prefixed) to bare pairs
# ("User: <chinese>" -> english), the format that bakes task-invariance
# into S0 (Preen-style). Output: data/zh2en_bare.json
import json
import re

PREFIX = re.compile(r"^(把这个句子翻译成英文：|Translate to English:)\s*")
src = json.load(open("data/zh2en_smoke.json", encoding="utf-8"))
out = []
for rec in src:
    bare = PREFIX.sub("", rec["instruction"]).strip()
    if not bare:
        continue
    out.append({"instruction": bare, "output": rec["output"]})
json.dump(out, open("data/zh2en_bare.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("pairs:", len(out))
print("sample:", json.dumps(out[0], ensure_ascii=False))
