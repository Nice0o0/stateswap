# One-off: audit README claims under the NEW sampling defaults (rep 1.25 +
# no_repeat_ngram 8). 翻译人格无指令英文输出率 + 猫娘风格命中率。
import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def call(path, method="GET", body=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


ZH = ["今天的月亮又圆又亮。", "孩子们在公园里放风筝。", "他因为堵车迟到了半个小时。",
      "这本书我已经读了三遍了。", "台风来了，航班全部取消。"]
s = call("/v1/sessions", "POST", {"persona": "zh2en-1.5b"})["session_id"]
print("=== zh2en-1.5b 无指令翻译（新默认参数）===")
ok = 0
for z in ZH:
    r = call("/v1/chat/completions", "POST", {
        "model": "zh2en-1.5b", "session_id": s,
        "messages": [{"role": "user", "content": z}], "temperature": 0.0})
    reply = r["choices"][0]["message"]["content"]
    letters = sum(c.isascii() and c.isalpha() for c in reply)
    cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in reply)
    good = letters >= 8 and cjk == 0
    ok += good
    print(f"{'✓' if good else '✗'} {z} -> {reply.strip()[:70]!r}")
print(f"English-output rate: {ok}/{len(ZH)}")

NEKO_MARKERS = ("喵", "主人", "本喵", "小鱼干", "（", "(")
PROMPTS = ["早上好呀！今天想吃小鱼干吗？", "陪我聊聊天吧，今天有点累", "给我讲个笑话吧",
           "我有点想你了", "晚饭吃点什么好呢？"]
s2 = call("/v1/sessions", "POST", {"persona": "neko-1.5b"})["session_id"]
print("\n=== neko-1.5b 风格命中（新默认参数）===")
hits = 0
for p in PROMPTS:
    r = call("/v1/chat/completions", "POST", {
        "model": "neko-1.5b", "session_id": s2,
        "messages": [{"role": "user", "content": p}], "temperature": 0.0})
    reply = r["choices"][0]["message"]["content"]
    good = any(m in reply for m in NEKO_MARKERS)
    hits += good
    print(f"{'✓' if good else '✗'} {p} -> {reply.strip()[:60]!r}")
print(f"Style hit-rate: {hits}/{len(PROMPTS)}")
