# One-off: build a unified neko-style corpus from ModelScope catgirl datasets.
# Sources (all Apache-2.0 per their repos, attribution in NOTICE.md):
#   kxdw2580/catgirl-datasets  v1/catgirl.json, v2/*.json
# Each record normalized to {"instruction", "output"}; <think> blocks stripped.
import json
import re
import os

SOURCES = {
    "catgirl_v1": r"%TEMP%\v1_catgirl.json",
    "catgirl_v2": r"%TEMP%\common_v2.json",
    "catgirl_v2_extra": r"%TEMP%\extra.json",
    "catgirl_neo1": r"%TEMP%\neo1.json",
    "catgirl_create": r"%TEMP%\create.json",
}
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def extract(rec):
    """Return (instruction, output) or None from any known format."""
    if not isinstance(rec, dict):
        return None
    if "instruction" in rec and "output" in rec:
        return str(rec["instruction"]).strip(), str(rec["output"]).strip()
    if "conversations" in rec:  # sharegpt-ish
        turns = rec["conversations"]
        if len(turns) >= 2:
            return str(turns[0].get("value", "")).strip(), str(turns[1].get("value", "")).strip()
    msgs = rec.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        return str(msgs[0].get("content", "")).strip(), str(msgs[1].get("content", "")).strip()
    if "prompt" in rec and "response" in rec:
        return str(rec["prompt"]).strip(), str(rec["response"]).strip()
    return None


seen = set()
out = []
stats = {}
for src, path in SOURCES.items():
    path = os.path.expandvars(path)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"skip {src}: {e!r}")
        continue
    if not isinstance(data, list):
        data = data.get("data", [])
    kept = 0
    for rec in data:
        pair = extract(rec)
        if not pair:
            continue
        inst, resp = pair
        resp = THINK.sub("", resp).strip()
        resp = resp.replace("\\~", "~").replace("\\*", "*")  # 源数据的 markdown 转义残留
        if not inst or not resp or len(resp) < 4:
            continue
        if len(inst) > 300 or len(resp) > 1200:
            continue
        key = inst
        if key in seen:
            continue
        seen.add(key)
        out.append({"instruction": inst, "output": resp})
        kept += 1
    stats[src] = (len(data), kept)

print("per-source (raw, kept):", stats)
print("total unique pairs:", len(out))
os.makedirs("data", exist_ok=True)
with open("data/neko_corpus_full.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
print("wrote data/neko_corpus_full.json")
print("sample:", json.dumps(out[123], ensure_ascii=False)[:220])
