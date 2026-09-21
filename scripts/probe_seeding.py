# One-off: 吸引子种子（attractor seeding）验证 v2。不是包的一部分。
#
# 假设（phase-transition.md E2b）：边界 α 处首回合交换把会话锁进对应吸引子。
# seed 把它产品化：新建会话预置一条人格种子交换（只写状态、不进 history）。
#
# v2 协议（v1 的两个教训）：
# 1) 纯 zh2en 在会话内 ~4 次交换后坍缩为空回复（"沉默型"退化，4-gram 护栏
#    不报）——全部测量在 ≤6 次交换内完成；
# 2) 无种子的会话内基线会被首探针自锁（"早上好呀"魔法效应）——补
#    equilibrium 臂（每探针新鲜会话）作为真正的无种子基线。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_phase_transition import ROOT, STYLE_SHORT, ZH_SHORT, Probe, save  # noqa: E402

OUT = ROOT / "docs" / "attractor-seeding.json"

STYLE_SEED = "喵呜~早呀主人！今天也要元气满满哦！(ฅ´ω`ฅ) 本喵的尾巴已经准备好啦～"
TASK_SEED = "Good morning! The moon was round and bright tonight."


def probe_one(p: Probe, persona: str, seed, in_session: bool = True) -> dict:
    """seeded/普通会话内：3 风格探针 → 3 任务探针（共 6 次交换）。"""
    sess = p.engine.new_session(persona, context_replay=0, seed=seed)
    try:
        hidden = (len(sess.history), sess.turns)
        style = p.style_probe(prompts=STYLE_SHORT, session_id=sess.session_id)
        task = p.task_probe(sentences=ZH_SHORT, session_id=sess.session_id)
        return {
            "persona": persona,
            "seed": seed if isinstance(seed, str) else ("auto" if seed else False),
            "hidden_history_len": hidden[0], "hidden_turns": hidden[1],
            "style_hit_rate": p.style_rate(style),
            "task_english_rate": p.task_rate(task),
            "replies": ([s["reply"] for s in style] + [t["reply"] for t in task]),
        }
    finally:
        p.engine.drop_session(sess.session_id)


def equilibrium(p: Probe, persona: str) -> dict:
    """无种子平衡态：每探针独立新鲜会话（真正的无种子水平）。"""
    style = p.style_probe(persona=persona, prompts=STYLE_SHORT)
    task = p.task_probe(persona=persona, sentences=ZH_SHORT)
    return {
        "persona": persona, "seed": "equilibrium(fresh-per-probe)",
        "style_hit_rate": p.style_rate(style),
        "task_english_rate": p.task_rate(task),
        "replies": [s["reply"] for s in style] + [t["reply"] for t in task],
    }


def main() -> None:
    p = Probe(str(ROOT / "models" / "rwkv7-1.5b-world-hf"),
              str(ROOT / "vendor" / "rwkv_vocab_v20230424.txt"))
    p.register_mix(0.5)  # 边界人格
    results = {"meta": {"boundary_alpha": 0.5,
                        "protocol": "greedy; 3 style + 3 task probes per arm; "
                                    "all measurements within 6 exchanges"}, "points": []}
    save(results, OUT)

    arms = [
        ("mix-0.50", "equilibrium", lambda: equilibrium(p, "mix-0.50")),
        ("mix-0.50", False, lambda: probe_one(p, "mix-0.50", False)),
        ("mix-0.50", "auto", lambda: probe_one(p, "mix-0.50", True)),
        ("mix-0.50", "style-text", lambda: probe_one(p, "mix-0.50", STYLE_SEED)),
        ("mix-0.50", "task-text", lambda: probe_one(p, "mix-0.50", TASK_SEED)),
        ("neko-1.5b", "equilibrium", lambda: equilibrium(p, "neko-1.5b")),
        ("zh2en-1.5b", "equilibrium", lambda: equilibrium(p, "zh2en-1.5b")),
    ]
    for persona, seed, fn in arms:
        row = fn()
        row["persona"], row["seed"] = persona, seed
        results["points"].append(row)
        save(results, OUT)
        print(f"{persona:<10} seed={str(seed):<24} style={row['style_hit_rate']:.2f} "
              f"task={row['task_english_rate']:.2f} | {row['replies'][0][:40]!r}")


if __name__ == "__main__":
    main()
