# One-off: S0 继承（人格热启动）实验评测（不是包的一部分）。
#
# 对照组：zh2en-1.5b-cold（S0=0 冷启动训 zh2en，2000 步）
# 实验组：zh2en-1.5b-warm（从 neko-1.5b-mt 热启动训同样的 zh2en，2000 步）
#
# 三个维度：
# 1. 任务：8 句裸中文 → 纯英文输出率（复用 probe_zh2en 的句表）
# 2. 风格：猫娘闲聊探针 → 风格标记命中率（donor neko-1.5b-mt 作参照）
# 3. 解剖：warm S0 与 donor / task 方向的逐层余弦——
#    验证人格解剖的预测：任务训练主要写前层，风格签名留在后层
#
# 结果写入 docs/inheritance.json（docs/persona-inheritance.md 引用）。
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.bench import NEKO_MARKERS, style_hit_rate  # noqa: E402
from stateswap.engine import Engine  # noqa: E402
from stateswap.s0 import load_s0_payload  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "rwkv7-1.5b-world-hf"
VOCAB = ROOT / "vendor" / "rwkv_vocab_v20230424.txt"

ZH = [
    "今天的月亮又圆又亮。",
    "孩子们在公园里放风筝。",
    "他因为堵车迟到了半个小时。",
    "这本书我已经读了三遍了。",
    "地铁上人太多了，我挤不上去。",
    "她用业余时间学会了弹吉他。",
    "台风来了，航班全部取消。",
    "程序员经常需要熬夜修 bug。",
]

STYLE_PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "你今天过得怎么样？",
    "我做了好多工作，好累啊",
    "周末我们去哪里玩？",
    "给我讲个笑话吧",
]


def is_english(reply: str) -> bool:
    """纯英文输出判定：有 ASCII 字母且不含 CJK 字符。"""
    has_letter = any(c.isascii() and c.isalpha() for c in reply)
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in reply)
    return has_letter and not has_cjk


def probe_translation(engine: Engine, persona: str) -> dict:
    """裸中文 → 英文率（贪心，独立会话）。"""
    replies = []
    for s in ZH:
        sess = engine.new_session(persona, context_replay=0)
        r = engine.chat(
            sess.session_id, s, max_new_tokens=64, temperature=0.0, top_p=1.0,
            rep_penalty=1.0, no_repeat_ngram=8,
        )["reply"]
        replies.append({"zh": s, "reply": r, "english": is_english(r)})
        engine.drop_session(sess.session_id)
    rate = sum(r["english"] for r in replies) / len(replies)
    return {"english_rate": rate, "samples": replies}


def probe_style(engine: Engine, persona: str) -> dict:
    """闲聊探针 → 猫娘风格命中（贪心，独立会话）。"""
    replies = []
    for p in STYLE_PROMPTS:
        sess = engine.new_session(persona, context_replay=0)
        r = engine.chat(
            sess.session_id, p, max_new_tokens=96, temperature=0.0, top_p=1.0,
            rep_penalty=1.0, no_repeat_ngram=8,
        )["reply"]
        replies.append({"prompt": p, "reply": r})
        engine.drop_session(sess.session_id)
    return {"style_hit_rate": style_hit_rate([r["reply"] for r in replies], NEKO_MARKERS),
            "samples": replies}


def layer_cosines(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """逐层逐头余弦均值：(L,H,64,64) → L 个标量。"""
    out = []
    for i in range(a.shape[0]):
        x = a[i].float().reshape(a.shape[1], -1)
        y = b[i].float().reshape(b.shape[1], -1)
        out.append(round(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean().item(), 4))
    return out


def main() -> None:
    engine = Engine(str(MODEL), str(VOCAB))
    arms = ["zh2en-1.5b", "zh2en-1.5b-cold", "zh2en-1.5b-warm", "zh2en-1.5b-warm-front"]
    for name in arms + ["neko-1.5b-mt"]:
        p = ROOT / "personas" / name / "s0.pt"
        if p.exists():
            engine.register_persona(name, p)

    result = {"model": str(MODEL), "steps": 2000, "recipe": "ctx=512 turns=1 lr=1e-4 seed=42"}

    for name in arms:
        if name not in engine.personas:
            continue
        print(f"== {name}: 翻译探针 ==")
        result.setdefault(name, {})["translation"] = probe_translation(engine, name)
        print(f"   english_rate = {result[name]['translation']['english_rate']:.0%}")

    for name in ["neko-1.5b-mt", "zh2en-1.5b-warm", "zh2en-1.5b-warm-front"]:
        if name not in engine.personas:
            continue
        print(f"== {name}: 风格探针 ==")
        result.setdefault(name, {})["style"] = probe_style(engine, name)
        print(f"   style_hit_rate = {result[name]['style']['style_hit_rate']:.0%}")

    # 逐层解剖：各继承臂与 donor（neko-mt）/ 任务方向（cold zh2en）的余弦
    names = [n for n in arms + ["neko-1.5b-mt"] if (ROOT / "personas" / n / "s0.pt").exists()]
    tensors = {
        name: load_s0_payload(torch.load(ROOT / "personas" / name / "s0.pt",
                                         map_location="cpu", weights_only=False))
        for name in names
    }
    anatomy = {}
    for arm in ["zh2en-1.5b-warm", "zh2en-1.5b-warm-front"]:
        if arm not in tensors:
            continue
        anatomy[f"cos_{arm}_vs_donor_neko"] = layer_cosines(tensors[arm], tensors["neko-1.5b-mt"])
        anatomy[f"cos_{arm}_vs_cold_zh2en"] = layer_cosines(tensors[arm], tensors["zh2en-1.5b-cold"])
    # 对照：donor 与冷启动任务人格本身近正交（state-arithmetic 的已知结论）
    anatomy["cos_donor_vs_cold"] = layer_cosines(tensors["neko-1.5b-mt"], tensors["zh2en-1.5b-cold"])
    L = next(iter(tensors.values())).shape[0]
    thirds = {}
    for key, arr in anatomy.items():
        thirds[key] = {
            "front(L0-7)": round(sum(arr[: L // 3]) / (L // 3), 4),
            "mid(L8-15)": round(sum(arr[L // 3: 2 * L // 3]) / (L // 3), 4),
            "back(L16-23)": round(sum(arr[2 * L // 3:]) / (L - 2 * (L // 3)), 4),
        }
        print(f"{key}: front={thirds[key]['front(L0-7)']} "
              f"mid={thirds[key]['mid(L8-15)']} back={thirds[key]['back(L16-23)']}")
    result["anatomy"] = {"per_layer": anatomy, "thirds": thirds}

    out = ROOT / "docs" / "inheritance.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
