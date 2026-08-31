# One-off: reproduce the WebUI's default sampling (temp 1.0 / top_p 0.9) chaos.
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


s = call("/v1/sessions", "POST", {"persona": "zh2en-1.5b"})["session_id"]
print("=== zh2en-1.5b @ UI 默认参数 (temp 1.0, top_p 0.9) ===")
for i, text in enumerate(["今天的月亮又圆又亮。", "孩子们在公园里放风筝。", "我明天要去北京出差。"]):
    r = call("/v1/chat/completions", "POST", {
        "model": "zh2en-1.5b", "session_id": s,
        "messages": [{"role": "user", "content": text}],
    })
    reply = r.get("choices", [{}])[0].get("message", {}).get("content", r)
    print(f"[{i}] {text}\n   -> {reply!r}")

print("\n=== neko-1.5b @ temp 1.0 连续 3 轮 ===")
s2 = call("/v1/sessions", "POST", {"persona": "neko-1.5b"})["session_id"]
for i, text in enumerate(["你好呀", "我养了一只狗，叫旺财", "你觉得它可爱吗？"]):
    r = call("/v1/chat/completions", "POST", {
        "model": "neko-1.5b", "session_id": s2,
        "messages": [{"role": "user", "content": text}],
    })
    reply = r.get("choices", [{}])[0].get("message", {}).get("content", r)
    print(f"[{i}] {text}\n   -> {reply!r}")
