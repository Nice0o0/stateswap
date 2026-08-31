# One-off: diagnose zh2en-1.5b translation quality (not part of the package).
# 顺带测： neko-1.5b 基线对话 / 会话接口。
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path, method="GET", body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode("utf-8", "replace")[:300]}


PROBES = [
    # (句子, 是否在训练集里)
    ("孩子们在公园里放风筝。", "IN colloquial"),
    ("今天的月亮又圆又亮。", "IN colloquial"),
    ("台风来了，航班全部取消。", "IN colloquial"),
    ("我们对这一结果表示谨慎的乐观。", "IN WMT news"),
    ("中方愿与各方一道，推动全球经济复苏。", "IN WMT news"),
    ("今天中午吃什么好呢？", "OUT casual"),
]

s = call("/v1/sessions", "POST", {"persona": "zh2en-1.5b"})["session_id"]
print("=== zh2en-1.5b greedy translation ===")
for text, tag in PROBES:
    r = call("/v1/chat/completions", "POST", {
        "model": "zh2en-1.5b", "session_id": s,
        "messages": [{"role": "user", "content": text}], "temperature": 0.0,
    })
    reply = r.get("choices", [{}])[0].get("message", {}).get("content", r)
    print(f"[{tag}] {text}\n   -> {reply!r}")
