# One-off: list files in RWKV/rwkv-7-world on ModelScope (not part of the package).
import json
import urllib.request

url = (
    "https://www.modelscope.cn/api/v1/models/RWKV/rwkv-7-world/repo/files"
    "?Revision=master&Root="
)
req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.load(r)

files = data["Data"]["Files"]
for f in files:
    print(f"{f.get('Size', 0):>14,}  LFS={f.get('IsLFS')}  {f.get('Path')}")
