# One-off: reproduce long-conversation degradation under realistic WebUI
# sampling (temp 0.7 / top_p 0.8, max_new 400) vs greedy.  Round 1 showed
# 12 greedy turns stay perfectly in persona — so drift-vs-S0 cosine does NOT
# predict failure.  This round hunts the real trigger: sampling temperature,
# long single replies, and snowballing state pollution across turns.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko", "personas/neko-1.5b/s0.pt")

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


def uniq_ratio(text: str, n: int = 4) -> float:
    """unique n-gram / total n-gram：复读越低。短文本返回 1。"""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def run(tag: str, temp: float, top_p: float, max_new: int) -> None:
    torch.manual_seed(0)
    s = engine.new_session("neko")
    print(f"\n===== {tag} (temp={temp}, top_p={top_p}, max_new={max_new}) =====")
    for t, p in enumerate(PROMPTS, 1):
        r = engine.chat(s.session_id, p, max_new_tokens=max_new,
                        temperature=temp, top_p=top_p)
        reply = r["reply"]
        hit = "Y" if any(m in reply for m in MARKERS) else "."
        uq = uniq_ratio(reply)
        flag = " <-- SUSPECT" if (uq < 0.6 or hit == ".") else ""
        print(f"T{t:02d} mk[{hit}] uq={uq:.2f} len={len(reply):3d} | {reply[:60]!r}{flag}")


run("WebUI 平衡档", 0.7, 0.8, 400)
run("贪心长回复", 0.0, 1.0, 400)
