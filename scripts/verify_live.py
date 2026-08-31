# One-off: live verification of the optimization pass (not part of the package).
# 1) streaming reply contains no U+FFFD even with kaomoji-heavy personas
# 2) concurrent request on the same session gets rejected (SessionBusy)
# 3) DELETE /v1/personas works
import json
import threading
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
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


# --- 1. streaming mojibake check ---
s = call("/v1/sessions", "POST", {"persona": "neko-0.4b-v2"})[1]
req = {"model": "neko-0.4b-v2", "session_id": s["session_id"], "stream": True,
       "messages": [{"role": "user", "content": "用颜文字跟我打个招呼！"}]}
with urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}, method="POST"), timeout=300) as r:
    raw = r.read().decode("utf-8")
reply = ""
for line in raw.splitlines():
    if line.startswith("data: ") and line != "data: [DONE]":
        p = json.loads(line[6:])
        d = p.get("choices", [{}])[0].get("delta", {}).get("content")
        if d:
            reply += d
print("1) streaming reply:", reply[:120])
print("   U+FFFD present:", "\ufffd" in reply, "| kaomoji chars:", sum(c in "ฅωﻌ^（~" for c in reply))

# --- 2. busy lock: hold the lock, then try to chat on the same session ---
# Lock via a raw generation: start a long stream in a thread, then second request should 409.
results = {}


def long_stream():
    try:
        req2 = dict(req, messages=[{"role": "user", "content": "给我讲一个很长很长的故事"}], max_new_tokens=200)
        with urllib.request.urlopen(urllib.request.Request(
                BASE + "/v1/chat/completions", data=json.dumps(req2).encode(),
                headers={"Content-Type": "application/json"}, method="POST"), timeout=300) as r:
            results["first"] = r.status
    except urllib.error.HTTPError as e:
        results["first"] = e.code


t = threading.Thread(target=long_stream)
t.start()
import time
time.sleep(4)  # 让第一个请求进入解码循环
code, body = call("/v1/chat/completions", "POST",
                  {"model": "neko-0.4b-v2", "session_id": s["session_id"],
                   "messages": [{"role": "user", "content": "并发测试"}]})
print("2) concurrent same-session status:", code, "(expect 409)")
t.join()

# --- 3. delete persona ---
code, _ = call("/v1/personas/mix-smoke", "DELETE")
print("3) DELETE mix-smoke (may 404 if absent):", code)
code, body = call("/v1/personas")
print("   personas now:", [p["name"] for p in body["personas"]])
