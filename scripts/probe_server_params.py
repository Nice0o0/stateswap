# One-off: A/B validation for the long-conversation fixes (not part of package).
# Run A 复现旧 server 行为的参数错位（rep_penalty=8.0）：预期前几轮就乱码。
# Run B 修复后参数 + 退化护栏：预期 15 轮全程干净；若有退化轮会被护栏
# 标记（degenerated=True）并回滚，后续轮次应恢复。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine, _unique_4gram_ratio  # noqa: E402

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


def run(tag: str, rep: float, guard: bool) -> None:
    torch.manual_seed(0)
    s = engine.new_session("neko")
    print(f"\n===== {tag} (rep_penalty={rep}, guard={guard}) =====")
    flagged = 0
    for t, p in enumerate(PROMPTS, 1):
        r = engine.chat(s.session_id, p, max_new_tokens=400,
                        temperature=0.7, top_p=0.8, rep_penalty=rep, guard=guard)
        reply = r["reply"]
        hit = "Y" if any(m in reply for m in MARKERS) else "."
        uq = _unique_4gram_ratio(reply)
        deg = "GUARD-ROLLBACK" if r.get("degenerated") else ""
        flagged += bool(r.get("degenerated"))
        sus = " <-- SUSPECT" if uq < 0.6 else ""
        print(f"T{t:02d} mk[{hit}] uq={uq:.2f} len={len(reply):3d} | {reply[:60]!r} {deg}{sus}")
    print(f"[{tag}] 护栏回滚次数: {flagged}")


run("A 旧参数复现", 8.0, False)
run("B 修复+护栏", 1.25, True)
