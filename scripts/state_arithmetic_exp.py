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
from stateswap.s0 import load_persona_file, load_s0_payload  # noqa: E402

MODEL = "models/rwkv7-0.4b-world-hf"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"
# 0.4B 人格已归档到 personas-legacy/（顶层 personas/ 只放 1.5B）
NEKO_PATH = "personas-legacy/neko-0.4b-v2/s0.pt"
ZH2EN_PATH = "personas-legacy/zh2en-0.4b-v3/s0.pt"

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
neko = load_s0_payload(load_persona_file(NEKO_PATH))
zh2en = load_s0_payload(load_persona_file(ZH2EN_PATH))


def _probe(persona: str, text: str, max_new: int) -> str:
    """单个探针 = 独立会话（context_replay=0）。

    v1 用两个长会话各跑 8 条探针——RWKV 的状态就是对话记忆，第 2 条起在
    续写第 1 条（源数据里 lerp_0.25/0.0 各 8 条回复完全相同），那测的是
    回声不是人格。v2 每题新开会话。"""
    sess = engine.new_session(persona, context_replay=0)
    try:
        return engine.chat(sess.session_id, text, max_new_tokens=max_new,
                           temperature=0.0, top_p=1.0, rep_penalty=1.0)["reply"]
    finally:
        engine.drop_session(sess.session_id)


def run(persona: str):
    style_replies = [_probe(persona, p, 96) for p in NEKO_PROMPTS]
    style_hits = [any(m in r for m in NEKO_MARKERS) for r in style_replies]
    trans_replies = [_probe(persona, s, 64).strip() for s in ZH_SENTENCES]
    trans_ok = []
    for r in trans_replies:
        letters = sum(c.isascii() and c.isalpha() for c in r)
        cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in r)
        trans_ok.append(letters >= 8 and cjk == 0)
    return {
        "style": sum(style_hits) / len(NEKO_PROMPTS),
        "translate": sum(trans_ok) / len(ZH_SENTENCES),
        "style_hits": style_hits,
        "english_flags": trans_ok,
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
    "config": {
        "neko": NEKO_PATH,
        "zh2en": ZH2EN_PATH,
        "decoding": "greedy",
        # v1 同会话连发（回声伪影）；v2 每题独立会话
        "protocol": "fresh session per probe, context_replay=0, rep_penalty=1.0, ngram=8",
        "protocol_version": 2,
        "note": ("v1（两个长会话各跑 8 条）里 lerp_0.25/lerp_0.0 的 8 条回复逐字相同，"
                 "属会话回声伪影；本文件为每个探针独立会话重测"),
    },
    "grid": {k: {"style": v["style"], "translate": v["translate"],
                 "style_hits": v["style_hits"], "english_flags": v["english_flags"]}
             for k, v in results.items()},
    "qualitative": qual,
    "replies": {k: {"style": v["style_replies"], "trans": v["trans_replies"]} for k, v in results.items()},
}
Path("docs").mkdir(exist_ok=True)
Path("docs/state_arithmetic.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved docs/state_arithmetic.json")
