# One-off: multi-turn-trained persona vs single-turn baseline on the same
# 15-turn long-conversation scenario (guard OFF — measures raw robustness).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine, _unique_4gram_ratio  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("st", "personas/neko-1.5b/s0.pt")
engine.register_persona("mt", "personas/neko-1.5b-mt/s0.pt")

MARKERS = ("喵", "主人", "本喵", "小鱼干", "ฅ", "（", "(")
PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "工作上被老板批评了，心情不太好",
    "给我详细讲讲你平时都在家里干什么，越详细越好",
    "然后呢？继续说说你的一天",
    "周末有什么好玩的地方推荐吗？详细介绍一下",
    "我最近在看一部科幻小说，关于时间旅行的",
    "时间旅行如果真的存在，你想去哪个时代？为什么？",
    "对了，法国的首都是哪里？",
    "那德国呢？顺便详细说说德国有什么美食",
    "给我写一篇400字的小故事，关于一只猫探险的",
    "这个故事的续集呢？再写400字",
    "谢谢你陪我聊这么久，总结一下我们今天聊了什么",
    "晚安啦，明天见",
    "再说一遍，法国的首都是哪里？",
]


def run(persona: str) -> None:
    torch.manual_seed(0)
    s = engine.new_session(persona)
    bad = 0
    print(f"\n===== {persona} (guard OFF, temp 0.7, max_new 400) =====")
    for t, p in enumerate(PROMPTS, 1):
        r = engine.chat(s.session_id, p, max_new_tokens=400,
                        temperature=0.7, top_p=0.8, rep_penalty=1.25, guard=False)
        reply = r["reply"]
        hit = any(m in reply for m in MARKERS)
        uq = _unique_4gram_ratio(reply)
        degenerate = len(reply) >= 200 and uq < 0.75
        bad += degenerate
        mark = "Y" if hit else "."
        sus = " <-- DEGENERATE" if degenerate else ""
        print(f"T{t:02d} mk[{mark}] uq={uq:.2f} len={len(reply):3d} | {reply[:60]!r}{sus}")
    print(f"[{persona}] 退化轮数: {bad}/15")


run("st")
run("mt")
