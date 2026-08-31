# Experiment: S0 arithmetic, round 2 (not part of the package).
# 1) 逐层混合：不同层用不同人格的 S0，检验能否绕过"全层同向混合"的相变
# 2) 方向减法：A - B（任务移除探针）
# 3) 相似度地图：各人格逐层余弦相似度
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

MODEL = "models/rwkv7-1.5b-world-hf"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"
NEKO_MARKERS = ("喵", "主人", "小鱼干", "宝宝", "本喵", "（", "(")
NEKO_PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "给我讲个笑话吧",
    "我有点想你了",
    "晚饭吃点什么好呢？",
    "你今天过得怎么样？",
]
ZH_SENTENCES = [
    "今天的月亮又圆又亮。",
    "孩子们在公园里放风筝。",
    "他因为堵车迟到了半个小时。",
    "这本书我已经读了三遍了。",
    "地铁上人太多了，我挤不上去。",
    "台风来了，航班全部取消。",
]

torch.manual_seed(11)
engine = Engine(MODEL, VOCAB)
neko = torch.load("personas/neko-1.5b/s0.pt", weights_only=False)["s0"].float()
zh2en = torch.load("personas/zh2en-1.5b/s0.pt", weights_only=False)["s0"].float()
L = neko.shape[0]


def make_mixed(pattern):
    """pattern: 每层一个 α（1=neko, 0=zh2en）"""
    alphas = torch.tensor(pattern, dtype=torch.float32).view(L, 1, 1, 1)
    return alphas * neko + (1 - alphas) * zh2en


def run_rates(name):
    s_style = engine.new_session(name)
    s_trans = engine.new_session(name)
    style_hits, trans_ok = 0, 0
    style_replies, trans_replies = [], []
    try:
        for p in NEKO_PROMPTS:
            r = engine.chat(s_style.session_id, p, max_new_tokens=96, temperature=0.0, top_p=1.0)["reply"]
            style_replies.append(r)
            if any(m in r for m in NEKO_MARKERS):
                style_hits += 1
        for s in ZH_SENTENCES:
            r = engine.chat(s_trans.session_id, s, max_new_tokens=64, temperature=0.0, top_p=1.0)["reply"].strip()
            trans_replies.append(r)
            letters = sum(c.isascii() and c.isalpha() for c in r)
            cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in r)
            if letters >= 8 and cjk == 0:
                trans_ok += 1
    finally:
        engine.drop_session(s_style.session_id)
        engine.drop_session(s_trans.session_id)
    return style_hits / len(NEKO_PROMPTS), trans_ok / len(ZH_SENTENCES), style_replies, trans_replies


mixtures = {
    "uniform_0.5": torch.full_like(neko, 0.5) * neko + torch.full_like(neko, 0.5) * zh2en,
    "front_neko": make_mixed([1.0] * (L // 2) + [0.0] * (L - L // 2)),
    "back_neko": make_mixed([0.0] * (L // 2) + [1.0] * (L - L // 2)),
    "alternate": make_mixed([1.0 if i % 2 == 0 else 0.0 for i in range(L)]),
    "sub_neko_minus_zh2en": neko - zh2en,
    "sub_zh2en_minus_neko": zh2en - neko,
}

results = {}
for name, tensor in mixtures.items():
    engine.register_tensor(name, tensor)
    s, t, sr, tr = run_rates(name)
    results[name] = {"style": s, "translate": t, "style_replies": sr, "trans_replies": tr}
    print(f"{name:>22}  style {s:>5.0%}  translate {t:>5.0%}")

# 相似度地图：逐层余弦相似度（mem-500 未训完时跳过该对）
mem500 = None
if Path("personas/mem-500/s0.pt").exists():
    mem500 = torch.load("personas/mem-500/s0.pt", weights_only=False)["s0"].float()
zero = torch.zeros_like(neko)


def layer_cos(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    return (torch.nn.functional.cosine_similarity(a, b, dim=1)).tolist()


sim = {
    "neko_vs_zh2en": layer_cos(neko, zh2en),
    "neko_vs_zero": layer_cos(neko, zero),
}
if mem500 is not None:
    sim["neko_vs_mem500"] = layer_cos(neko, mem500)
    sim["zh2en_vs_mem500"] = layer_cos(zh2en, mem500)
print("neko↔zh2en per-layer cos:", [f"{x:.2f}" for x in sim["neko_vs_zh2en"]])

out = {"config": {"L": L, "poles": ["neko-1.5b", "zh2en-1.5b"], "decoding": "greedy"},
       "mixtures": {k: {"style": v["style"], "translate": v["translate"]} for k, v in results.items()},
       "qualitative": {k: {"style": v["style_replies"][0], "translate": v["trans_replies"][0]}
                       for k, v in results.items()},
       "similarity": sim,
       "replies": results}
Path("docs").mkdir(exist_ok=True)
Path("docs/state_arithmetic2.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved docs/state_arithmetic2.json")
