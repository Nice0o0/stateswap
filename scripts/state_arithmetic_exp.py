# Experiment: S0 arithmetic — do persona states compose linearly?
# 插值 α·S_neko + (1-α)·S_zh2en 与相加混合，贪心解码（温度 0）消除采样运气。
# 指标：
#   style     — 8 个闲聊 prompt 的猫娘标记词命中率
#   translate — 8 个裸中文句子的英文输出率（无任何指令）
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.arithmetic import add, interpolate  # noqa: E402
from stateswap.engine import Engine  # noqa: E402

MODEL = "models/rwkv7-0.4b-world-hf"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"
NEKO_PATH = "personas/neko-0.4b-v2/s0.pt"
ZH2EN_PATH = "personas/zh2en-0.4b-v3/s0.pt"

NEKO_MARKERS = ("喵", "主人", "小鱼干", "宝宝", "本喵", "（", "(")
NEKO_PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "周末我们去哪里玩？",
    "给我讲个笑话吧",
    "我有点想你了",
    "晚饭吃点什么好呢？",
    "你今天过得怎么样？",
    "夸夸我吧！",
]
ZH_SENTENCES = [
    "今天的月亮又圆又亮。",
    "孩子们在公园里放风筝。",
    "他因为堵车迟到了半个小时。",
    "这本书我已经读了三遍了。",
    "地铁上人太多了，我挤不上去。",
    "她用业余时间学会了弹吉他。",
    "台风来了，航班全部取消。",
    "程序员经常需要熬夜修 bug。",
]

torch.manual_seed(11)
engine = Engine(MODEL, VOCAB)
neko = torch.load(NEKO_PATH, weights_only=False)["s0"].float()
zh2en = torch.load(ZH2EN_PATH, weights_only=False)["s0"].float()


def run(persona: str):
    """风格与翻译各用独立会话：会话状态包含对话历史，混用会污染
    （历史里的猫娘对话会让"翻译"探测变成"继续聊天"）。"""
    style_hits = 0
    style_replies = []
    trans_ok = 0
    trans_replies = []
    style_session = engine.new_session(persona)
    trans_session = engine.new_session(persona)
    try:
        for p in NEKO_PROMPTS:
            r = engine.chat(style_session.session_id, p, max_new_tokens=96, temperature=0.0, top_p=1.0)["reply"]
            style_replies.append(r)
            if any(m in r for m in NEKO_MARKERS):
                style_hits += 1
        for s in ZH_SENTENCES:
            r = engine.chat(trans_session.session_id, s, max_new_tokens=64, temperature=0.0, top_p=1.0)["reply"].strip()
            trans_replies.append(r)
            letters = sum(c.isascii() and c.isalpha() for c in r)
            cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in r)
            if letters >= 8 and cjk == 0:
                trans_ok += 1
    finally:
        engine.drop_session(style_session.session_id)
        engine.drop_session(trans_session.session_id)
    return {
        "style": style_hits / len(NEKO_PROMPTS),
        "translate": trans_ok / len(ZH_SENTENCES),
        "style_replies": style_replies,
        "trans_replies": trans_replies,
    }


mixtures = {}
for a in [1.0, 0.75, 0.5, 0.25, 0.0]:
    mixtures[f"lerp_{a}"] = interpolate(neko, zh2en, a)
mixtures["sum_half"] = add(neko, zh2en, 0.5)
mixtures["sum_raw"] = neko + zh2en

for name, tensor in mixtures.items():
    engine.register_tensor(name, tensor)

results = {name: run(name) for name in mixtures}
for name, r in results.items():
    print(f"{name:>10}  style {r['style']:>5.0%}  translate {r['translate']:>5.0%}")

print("\n--- 定性样例（lerp_0.5 与 sum_raw）---")
qual = {}
for name in ["lerp_0.5", "sum_raw"]:
    qual[name] = {
        "style": results[name]["style_replies"][0],
        "translate": results[name]["trans_replies"][0],
    }
    print(f"[{name}][style]     {results[name]['style_replies'][0]!r}")
    print(f"[{name}][translate] {results[name]['trans_replies'][0]!r}")

out = {
    "config": {"neko": NEKO_PATH, "zh2en": ZH2EN_PATH, "decoding": "greedy"},
    "grid": {k: {"style": v["style"], "translate": v["translate"]} for k, v in results.items()},
    "qualitative": qual,
    "replies": {k: {"style": v["style_replies"], "translate": v["trans_replies"]} for k, v in results.items()},
}
Path("docs").mkdir(exist_ok=True)
Path("docs/state_arithmetic.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved docs/state_arithmetic.json")
