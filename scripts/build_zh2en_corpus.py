# One-off: build zh->en translation corpus from WMT newstest (ModelScope,
# iic/WMT-Chinese-to-English-Machine-Translation-newstest) + colloquial pairs.
# Output: data/zh2en_corpus_full.json (~5000 bare {instruction: zh, output: en})
import csv
import json
import os
import random

rows = []
for n in ["wmt18", "wmt19", "wmt20"]:
    p = os.path.expandvars(r"%TEMP%\{}.csv".format(n))
    for r in list(csv.reader(open(p, encoding="utf-8")))[1:]:
        if len(r) >= 2:
            rows.append((r[0].strip(), r[1].strip()))
print("wmt raw:", len(rows))

seen = set()
pairs = []
for zh, en in rows:
    if not zh or not en or zh in seen:
        continue
    if len(zh) > 220 or len(en) > 500 or len(en) < 8:
        continue
    seen.add(zh)
    pairs.append({"instruction": zh, "output": en})

colloquial = json.load(open("data/zh2en_bare.json", encoding="utf-8"))
have = {p["instruction"] for p in pairs}
for p in colloquial:
    if p["instruction"] not in have:
        pairs.append(p)

random.Random(7).shuffle(pairs)
print("total pairs:", len(pairs))
with open("data/zh2en_corpus_full.json", "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=0)
print("wrote data/zh2en_corpus_full.json")
print("sample:", json.dumps(pairs[0], ensure_ascii=False)[:200])
