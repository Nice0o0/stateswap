# One-off: anatomy follow-up — find the rank cliff, and validate rank-4
# beyond 5 prompts (15-turn long conversation + knowledge retention).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine, _unique_4gram_ratio  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko", "personas/neko-1.5b/s0.pt")
engine.register_persona("zh2en", "personas/zh2en-1.5b/s0.pt")
neko_s0 = engine.personas["neko"].s0.cpu()
zh_s0 = engine.personas["zh2en"].s0.cpu()

NEKO_MARKERS = ("喵", "主人", "本喵", "小鱼干", "ฅ", "（", "(")
NEKO_PROMPTS = ["早上好呀！今天想吃小鱼干吗？", "陪我聊聊天吧，今天有点累",
                "给我讲个笑话吧", "我有点想你了", "晚饭吃点什么好呢？"]
ZH_PROMPTS = ["今天的月亮又圆又亮。", "孩子们在公园里放风筝。", "他因为堵车迟到了半个小时。",
              "这本书我已经读了三遍了。", "台风来了，航班全部取消。"]
KNOWLEDGE = ["法国的首都是哪里？", "水的化学式是什么？", "一年有多少个月？"]
LONG15 = [
    "早上好呀！今天想吃小鱼干吗？", "陪我聊聊天吧，今天有点累", "工作上被老板批评了，心情不太好",
    "给我详细讲讲你平时都在家里干什么，越详细越好", "然后呢？继续说说你的一天",
    "周末有什么好玩的地方推荐吗？详细介绍一下", "我最近在看一部科幻小说，关于时间旅行的",
    "时间旅行如果真的存在，你想去哪个时代？为什么？", "对了，法国的首都是哪里？",
    "那德国呢？顺便详细说说德国有什么美食", "给我写一篇400字的小故事，关于一只猫探险的",
    "这个故事的续集呢？再写400字", "谢谢你陪我聊这么久，总结一下我们今天聊了什么",
    "晚安啦，明天见", "再说一遍，法国的首都是哪里？",
]


def truncate(s0: torch.Tensor, k: int) -> torch.Tensor:
    L_, H = s0.shape[0], s0.shape[1]
    u, s, vh = torch.linalg.svd(s0.float().reshape(L_ * H, 64, 64))
    rec = (u[:, :, :k] * s[:, None, :k]) @ vh[:, :k, :]
    return rec.reshape(L_, H, 64, 64)


def chat(persona, text, **kw):
    s = engine.new_session(persona)
    args = dict(max_new_tokens=64, temperature=0.0, top_p=1.0,
                rep_penalty=1.0, no_repeat_ngram=0, guard=False)
    args.update(kw)
    r = engine.chat(s.session_id, text, **args)
    engine.drop_session(s.session_id)
    return r["reply"]


def neko_rate(persona):
    return sum(any(m in chat(persona, p) for m in NEKO_MARKERS) for p in NEKO_PROMPTS) / len(NEKO_PROMPTS)


def zh_rate(persona):
    ok = 0
    for z in ZH_PROMPTS:
        reply = chat(persona, z)
        letters = sum(c.isascii() and c.isalpha() for c in reply)
        cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in reply)
        ok += letters >= 8 and cjk == 0
    return ok / len(ZH_PROMPTS)


out = {"cliff": {}, "rank4_longconv": [], "rank4_knowledge": []}

# 1) 找悬崖：rank-3/2/1
for k in (3, 2, 1):
    engine.register_tensor(f"probe-nr{k}", truncate(neko_s0, k))
    engine.register_tensor(f"probe-zr{k}", truncate(zh_s0, k))
    nr, zr = neko_rate(f"probe-nr{k}"), zh_rate(f"probe-zr{k}")
    out["cliff"][f"rank{k}"] = {"neko": nr, "zh2en": zr}
    print(f"[cliff] rank-{k}: neko={nr:.0%} zh2en={zr:.0%}")
    engine.delete_persona(f"probe-nr{k}")
    engine.delete_persona(f"probe-zr{k}")

# 2) rank-4 的 15 轮长对话（guard OFF）
engine.register_tensor("neko-r4", truncate(neko_s0, 4))
torch.manual_seed(0)
s = engine.new_session("neko-r4")
bad = 0
for t, p in enumerate(LONG15, 1):
    r = engine.chat(s.session_id, p, max_new_tokens=400, temperature=0.7,
                    top_p=0.8, rep_penalty=1.25, guard=False)
    uq = _unique_4gram_ratio(r["reply"])
    deg = len(r["reply"]) >= 200 and uq < 0.75
    bad += deg
    out["rank4_longconv"].append({"turn": t, "uq": round(uq, 2), "len": len(r["reply"]), "degenerate": deg})
    sus = " <-- DEGENERATE" if deg else ""
    print(f"[r4-long] T{t:02d} uq={uq:.2f} len={len(r['reply']):3d} | {r['reply'][:56]!r}{sus}")
print(f"[r4-long] 退化轮数: {bad}/15")
engine.drop_session(s.session_id)

# 3) rank-4 知识保持（底座能力应不受影响）
for q in KNOWLEDGE:
    reply = chat("neko-r4", q, max_new_tokens=80)
    out["rank4_knowledge"].append({"q": q, "reply": reply[:80]})
    print(f"[r4-know] {q} -> {reply[:60]!r}")
engine.delete_persona("neko-r4")

Path("docs/anatomy2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved docs/anatomy2.json")
