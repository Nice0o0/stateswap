# One-off: 风格烘焙数据 scaling 曲线（风格命中率 + 能力保持 vs 语料量）。
# 不是包的一部分。协议：贪心、独立会话、context_replay=0。
#
# 数据点：
#   scaling-neko-{50..1600}（本轮：N 对语料 × 800 步，seed 42）
#   neko-1.5b-400（3095 对 × 400 步）/ neko-1.5b（3095 对 × 4000 步）—— steps 轴
# 能力保持：factory 的 8 条中性事实探针（期望答案子串匹配）——把 three_way
# 发现（深度注入覆盖通用能力）量化成第二条曲线。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_phase_transition import ROOT, Probe, save  # noqa: E402

from stateswap.factory import NEUTRAL_PROBES  # noqa: E402

OUT = ROOT / "docs" / "scaling.json"

POINTS = [
    # (persona, pairs, steps) —— size × steps 网格
    ("scaling-neko-50", 50, 800),
    ("scaling-neko-100", 100, 800),
    ("scaling-neko-200", 200, 800),
    ("scaling-neko-400", 400, 800),
    ("scaling-neko-800", 800, 800),
    ("scaling-neko-1600", 1600, 800),
    ("neko-1.5b-400", 3095, 400),
    ("neko-1.5b", 3095, 4000),
    # steps 轴（3095 对）
    ("scaling-steps-100", 3095, 100),
    ("scaling-steps-200", 3095, 200),
    # 200 步下的 size 轴
    ("scaling-50st200", 50, 200),
    ("scaling-400st200", 400, 200),
]


def main() -> None:
    p = Probe(str(ROOT / "models" / "rwkv7-1.5b-world-hf"),
              str(ROOT / "vendor" / "rwkv_vocab_v20230424.txt"))
    results = {"meta": {"protocol": "greedy, fresh sessions, context_replay=0, seed 42",
                        "steps_fixed": 800}, "points": []}
    save(results, OUT)
    for name, pairs, steps in POINTS:
        path = ROOT / "personas" / name / "s0.pt"
        if not path.exists():
            print(f"skip {name} (no s0.pt)")
            continue
        p.engine.register_persona(name, path)
        style = p.style_probe(persona=name)
        # 能力保持：逐题独立会话（与 factory eval 同口径）
        neutral = []
        for q, expected in NEUTRAL_PROBES:
            sess = p.engine.new_session(name, context_replay=0)
            try:
                r = p.engine.chat(sess.session_id, q, max_new_tokens=96,
                                  temperature=0.0, top_p=1.0,
                                  rep_penalty=1.0, no_repeat_ngram=8)
            finally:
                p.engine.drop_session(sess.session_id)
            neutral.append({"prompt": q, "expected": expected, "reply": r["reply"][:80],
                            "correct": expected.lower() in r["reply"].lower()})
        row = {
            "persona": name, "pairs": pairs, "steps": steps,
            "style_hit_rate": p.style_rate(style),
            "neutral_correct_rate": sum(x["correct"] for x in neutral) / len(neutral),
            "degenerated": sum(s["degenerated"] for s in style),
            "style": style, "neutral": neutral,
        }
        results["points"].append(row)
        save(results, OUT)
        print(f"{name} ({pairs} pairs, {steps} steps): style={row['style_hit_rate']:.2f} "
              f"neutral={row['neutral_correct_rate']:.2f} degen={row['degenerated']}")


if __name__ == "__main__":
    main()
