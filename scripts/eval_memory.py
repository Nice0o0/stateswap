# Experiment: S0 compressed-memory evaluation (not part of the package).
# 对每个容量档的 S0 与各基线，回答记忆保持/泛化/能力保持三组问题。
#
# 基线：
#   S0=0            —— 底座裸奔（幻觉率参照）
#   RAG-oracle      —— 把金标准事实放进 prompt（检索上界）
#   S0@{100,300,500} —— 压缩记忆
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

MODEL = "models/rwkv7-1.5b-world-hf"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"
GENERAL_PROBES = [  # 能力保持探针：S0 注入知识后，底座通用行为是否被破坏
    "What is the capital of France?",
    "把这句话翻译成英文：孩子们在公园里放风筝。",
    "用一句话介绍你自己。",
    "1 加 1 等于几？",
]

_norm = re.compile(r"[\s，。？！,.?!、；;：:'\"“”‘’（）()]")


def norm(s: str) -> str:
    return _norm.sub("", s)


def grade(reply: str, gold: str) -> bool:
    """规范化后，金标准值出现在回复中即算对（容忍前缀/后缀噪音）。"""
    g, r = norm(gold), norm(reply)
    return bool(g) and g in r


def ask(engine, session_id, question, temperature=0.0):
    r = engine.chat(session_id, question, max_new_tokens=48, temperature=temperature, top_p=1.0)
    return r["reply"].strip()


def run_eval(engine, eval_set, persona, rag_facts=None, tag=""):
    """rag_facts: RAG-oracle 模式下注入的金标准事实（None 则纯问答）。"""
    session = engine.new_session(persona)
    results = []
    try:
        for q in eval_set:
            if rag_facts is not None:
                fact = next(f for f in rag_facts if f[0] == q["subject"] and f[1] == q["attribute"])
                text = (f"参考信息：{q['subject']}的{q['attribute']}是{fact[2]}。"
                        f"请仅根据参考信息回答问题：{q['instruction']}")
            else:
                text = q["instruction"]
            reply = ask(engine, session.session_id, text)
            results.append({"q": q["instruction"], "gold": q["output"],
                            "reply": reply, "correct": grade(reply, q["output"])})
    finally:
        engine.drop_session(session.session_id)
    acc = sum(r["correct"] for r in results) / max(1, len(results))
    print(f"  [{tag}] accuracy {acc:.0%} ({sum(r['correct'] for r in results)}/{len(results)})")
    return {"accuracy": acc, "detail": results}


def main():
    engine = Engine(MODEL, VOCAB)
    eval_data = json.loads(Path("data/knowledge_eval.json").read_text(encoding="utf-8"))
    mem_set = eval_data["memorization"]
    gen_set = eval_data["generalization"]
    mem_facts = eval_data["memorization_facts"]
    out = {"model": MODEL}

    # 1) S0=0 基线：无知识裸奔
    out["baseline_s0"] = {"memorization": run_eval(engine, mem_set, "none", tag="s0=0 mem"),
                          "generalization": run_eval(engine, gen_set, "none", tag="s0=0 gen")}

    # 2) RAG-oracle：金标准事实进 prompt（检索上界）
    out["rag_oracle"] = run_eval(engine, mem_set, "none", rag_facts=mem_facts, tag="rag mem")

    # 3) 各容量档的压缩记忆
    for size in (100, 300, 500):
        name = f"mem-{size}"
        s0_path = Path(f"personas/{name}/s0.pt")
        if not s0_path.exists():
            print(f"  [skip] {name} not trained")
            continue
        engine.register_persona(name, str(s0_path))
        out[name] = {"memorization": run_eval(engine, mem_set, name, tag=f"{name} mem"),
                     "generalization": run_eval(engine, gen_set, name, tag=f"{name} gen")}

    # 4) 能力保持：mem-500 加载后回答通用问题
    if Path("personas/mem-500/s0.pt").exists():
        engine.register_persona("mem-500", "personas/mem-500/s0.pt")
        session = engine.new_session("mem-500")
        out["general_probes"] = []
        try:
            for q in GENERAL_PROBES:
                out["general_probes"].append(
                    {"q": q, "reply": ask(engine, session.session_id, q, temperature=0.0)})
        finally:
            engine.drop_session(session.session_id)
        for gp in out["general_probes"]:
            print(f"  [general] {gp['q']} -> {gp['reply']!r:.100}")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/compressed_memory.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved docs/compressed_memory.json")


if __name__ == "__main__":
    main()
