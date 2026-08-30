# One-off network probe for ModelScope repo candidates (not part of the package).
import json
import urllib.request

CANDIDATES = [
    "BlinkDL/rwkv-7-world",
    "RWKV/rwkv-7-world",
    "AI-ModelScope/rwkv-7-world",
    "RWKV/rwkv7-world",
    "BlinkDL/rwkv7-world",
]


def probe(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read(400).decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        return "ERR", str(e)[:160]


for path in CANDIDATES:
    status, body = probe(f"https://www.modelscope.cn/api/v1/models/{path}")
    print(path, "->", status, repr(body[:200]))

# Also try the generic model-files API shape on the most likely candidate.
print("\n--- files API probe ---")
for path in CANDIDATES:
    url = f"https://www.modelscope.cn/api/v1/models/{path}/repo/files?Revision=master&Root="
    status, body = probe(url)
    print(path, "->", status, repr(body[:200]))
