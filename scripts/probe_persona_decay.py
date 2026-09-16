# One-off: persona decay over multi-turn conversation (not part of package).
# 假设：S0 只在"对话开头"被训练（每条样本独立冷启动），多轮后递归状态漂移
# 出训练分布 → 人格消退、底座漂移成胡言乱语。
# 实验：同一引擎跑两轮同样的 12 轮对话——对照组 vs 每轮回锚（S ← (1-β)S + βS0）。
# 每轮记录：人格标记命中率、当前状态与 S0 的逐层余弦相似度、回复长度。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko", "personas/neko-1.5b/s0.pt")
L = engine.model.config.num_hidden_layers

MARKERS = ("喵", "主人", "本喵", "小鱼干", "ฅ", "（", "(")
PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "工作上被老板批评了，心情不太好",
    "你觉得我该怎么调整心态呢？",
    "讲个笑话让我开心一下吧",
    "周末有什么好玩的地方推荐吗？",
    "我最近在看一部科幻小说，关于时间旅行的",
    "时间旅行如果真的存在，你想去哪个时代？",
    "对了，法国的首都是哪里？",
    "那德国呢？顺便说说德国有什么美食",
    "谢谢你陪我聊这么久，总结一下吧",
    "晚安啦，明天见",
]


def state_cos(session) -> float:
    """当前各层 recurrent_state 与人格 S0 的平均逐头余弦相似度。"""
    ref = engine.personas[session.persona_name].s0  # (L,H,64,64)
    sims = []
    for i in range(L):
        cur = session.cache[i]["recurrent_state"][0].float().reshape(ref.shape[1], -1)
        r = ref[i].float().reshape(ref.shape[1], -1)
        sims.append(torch.nn.functional.cosine_similarity(cur, r, dim=-1).mean().item())
    return sum(sims) / len(sims)


def reanchor(session, beta: float) -> None:
    """把当前递归状态向 S0 回拉 β 比例（模拟'人格再注入'）。"""
    ref = engine.personas[session.persona_name].s0
    with torch.no_grad():
        for i in range(L):
            st = session.cache[i]["recurrent_state"]
            st.copy_((1 - beta) * st.float() + beta * ref[i].unsqueeze(0).to(st.device))


def run(beta: float) -> None:
    torch.manual_seed(0)
    s = engine.new_session("neko")
    tag = "对照组" if beta == 0 else f"回锚 β={beta}"
    print(f"\n===== {tag} =====")
    for t, p in enumerate(PROMPTS, 1):
        r = engine.chat(s.session_id, p, max_new_tokens=128,
                        temperature=0.0, top_p=1.0)
        if beta:
            reanchor(s, beta)
        reply = r["reply"]
        hit = any(m in reply for m in MARKERS)
        cos = state_cos(s)
        mark = "Y" if hit else "."
        print(f"T{t:02d} marker[{mark}] cos={cos:.3f} "
              f"len={len(reply):3d} | {reply[:48]!r}")


run(0.0)
run(0.15)
