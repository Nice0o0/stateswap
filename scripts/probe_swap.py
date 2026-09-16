# One-off: what each persona actually does with the same input, and whether
# conversation stays clean after hot-swapping (requires a running server).
import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path, method="GET", body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(persona, text, session=None):
    body = {"model": persona, "messages": [{"role": "user", "content": text}],
            "temperature": 0.7, "max_tokens": 96}
    if session:
        body["session_id"] = session
    r = call("/v1/chat/completions", "POST", body)
    return r["choices"][0]["message"]["content"].strip()


P = "你好呀，陪我聊聊天吧"
print("===== 同一输入，不同人格（新会话） =====")
for persona in ("none", "neko-1.5b-mt", "zh2en-1.5b", "mem-100"):
    print(f"\n[{persona}] {chat(persona, P)[:80]!r}")

print("\n===== 同一会话连续热切换（每次切换后立即对话） =====")
sid = call("/v1/sessions", "POST", {"persona": "neko-1.5b-mt"})["session_id"]
print(f"[neko-1.5b-mt] {chat('neko-1.5b-mt', P, sid)[:80]!r}")
for target in ("zh2en-1.5b", "none", "neko-1.5b-rank4", "neko-1.5b-mt"):
    r = call(f"/v1/sessions/{sid}/swap", "POST", {"persona": target})
    reply = chat(target, P, sid)
    print(f"[swap->{target} ({r['latency_ms']:.1f}ms)] {reply[:80]!r}")
call(f"/v1/sessions/{sid}", "DELETE")
