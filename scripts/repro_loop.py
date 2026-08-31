# One-off: reproduce the user's adversarial multi-turn scenario (not part of package).
# 复现用户报告的对话循环：短消息连发 + 对抗性输入，验证早停+重复惩罚后不再循环。
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
        return {"error": e.read().decode("utf-8", "replace")[:200]}


s = call("/v1/sessions", "POST", {"persona": "neko-1.5b"})["session_id"]
TURNS = [
    "你好",
    "你是谁",
    "我不是猫娘",
    "要",
    "要",
    "不要",
    "喜欢",
    "不喜欢",
    "你是谁",
    "你好",
]
print("=== neko-1.5b @ 平衡档(0.7) 多轮对抗复现 ===")
for t in TURNS:
    r = call("/v1/chat/completions", "POST", {
        "model": "neko-1.5b", "session_id": s,
        "messages": [{"role": "user", "content": t}],
    })
    reply = r.get("choices", [{}])[0].get("message", {}).get("content", r)
    print(f"你: {t}\nS: {reply.strip()[:90]!r}\n")

# 循环检测：相邻回复两两比较，统计完全相同/高度重合对
hist = call(f"/v1/sessions/{s}").get("history", [])
replies = [m["content"] for m in hist if m["role"] == "assistant"]
dup = sum(1 for a, b in zip(replies, replies[1:]) if a == b)
print(f"相邻回复完全相同对: {dup}/{len(replies)-1}")
