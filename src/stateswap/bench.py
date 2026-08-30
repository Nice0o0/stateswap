"""Benchmarks for the stateswap serving stack.

产出三组数字（docs/benchmarks.md）：
1. 行为：S0 人格 vs S0=0 基线的风格命中率（猫娘标记词）与上下文成本。
2. 服务：人格热切换延迟、每会话状态内存、解码速度。
3. 架构：session 模式（O(1) 状态续聊） vs 无状态模式（每请求重放历史）
   在多轮对话下的 prefill 延迟增长。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .engine import Engine

NEKO_MARKERS = ("喵", "主人", "（", "(", "小鱼干", "宝宝")


def style_hit_rate(replies: list[str], markers: tuple[str, ...]) -> float:
    hits = sum(1 for r in replies if any(m in r for m in markers))
    return hits / max(1, len(replies))


def bench_style(engine: Engine, prompts: list[str], persona: str, max_new: int = 96) -> dict:
    session = engine.new_session(persona)
    replies = []
    try:
        for p in prompts:
            result = engine.chat(session.session_id, p, max_new_tokens=max_new, temperature=1.0)
            replies.append(result["reply"])
    finally:
        engine.drop_session(session.session_id)
    return {"replies": replies, "persona": persona}


def bench_serving(engine: Engine, persona_a: str, persona_b: str, turns: int = 12) -> dict:
    out: dict = {}

    # 1) persona swap latency
    session = engine.new_session(persona_a)
    lat = []
    for name in [persona_b, persona_a, persona_b]:
        lat.append(engine.swap_persona(session.session_id, name)["latency_ms"])
    out["swap_latency_ms"] = lat
    out["session_memory_mb"] = round(session.memory_mb(engine.model), 3)
    out["persona_size_mb"] = round(engine.personas[persona_a].size_mb, 3)

    # 2) decode speed (single token loop) — reuse session, feed fixed prompt
    t0 = time.perf_counter()
    result = engine.chat(session.session_id, "介绍一下你自己。", max_new_tokens=128, temperature=1.0)
    out["decode_ms_per_token_128tok"] = result["decode_ms_per_token"]
    out["decode_wall_s_128tok"] = round(time.perf_counter() - t0, 2)
    engine.drop_session(session.session_id)

    # 3) session-mode vs stateless-mode prefill latency as history grows
    sess = engine.new_session(persona_a)
    session_prefill = []
    stateless_prefill = []
    utterances = [
        "我们继续聊聊吧", "你今天过得怎么样？", "给我讲个小故事", "再来一个短的",
        "说说你的爱好", "你喜欢的食物是什么？", "今天天气不错对吧", "推荐一部电影",
        "周末有什么计划？", "最近在读什么书？", "给我出个谜语", "再讲一个笑话",
    ][:turns]
    for i, u in enumerate(utterances):
        result = engine.chat(sess.session_id, u, max_new_tokens=24, temperature=1.0)
        session_prefill.append(result["prefill_ms"])
        # stateless: 从零重放全部历史（i 轮 user+assistant 对 + 新输入）
        engine.complete_from_messages(
            persona_a,
            [{"role": "user", "content": x} for x in utterances[: i]] + [{"role": "user", "content": u}],
            max_new_tokens=1,
        )
        stateless_prefill.append(None)  # 延迟在下方用 prefill-only 测量
    engine.drop_session(sess.session_id)
    out["session_prefill_ms_by_turn"] = session_prefill
    out["stateless_turns"] = turns
    return out


def bench_prefill_only(engine: Engine, persona: str, turn_tokens: list[int]) -> dict:
    """无状态模式下每请求需 prefill 的 token 数与耗时（真实重放）。"""
    session = engine.new_session(persona)
    cache = engine._cache_for(engine.personas[persona])
    rows = []
    filler = "关于这个话题我们再展开聊聊，多给一些细节和例子。"
    for t in turn_tokens:
        # 构造约 t 个 token 的历史
        reps = max(1, t // max(1, len(engine.tok.encode("User: " + filler + "\n\n"))))
        history = ("User: " + filler + "\n\n") * reps
        ids = engine.tok.encode(history)
        t0 = time.perf_counter()
        with torch.no_grad():
            engine.model(
                input_ids=torch.tensor([ids], device=engine.device),
                past_key_values=cache,
                use_cache=True,
            )
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"history_tokens": len(ids), "prefill_ms": round(ms, 1)})
    engine.drop_session(session.session_id)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")
    ap.add_argument("--persona-dir", default="personas")
    ap.add_argument("--persona", default="neko-0.4b")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--out", default="docs/benchmarks.json")
    args = ap.parse_args()

    engine = Engine(args.model, args.vocab)
    for p in sorted(Path(args.persona_dir).glob("*/s0.pt")):
        engine.register_persona(p.parent.name, str(p))

    style_prompts = [
        "早上好呀！今天想吃小鱼干吗？",
        "陪我聊聊天吧，今天有点累",
        "你今天过得怎么样？",
        "我做了好多工作，好累啊",
        "周末我们去哪里玩？",
        "给我讲个笑话吧",
        "我有点想你了",
        "晚饭吃点什么好呢？",
    ]
    result = {"model": args.model, "persona": args.persona}

    base = bench_style(engine, style_prompts, "none")
    personad = bench_style(engine, style_prompts, args.persona)
    result["style"] = {
        "baseline_hit_rate": style_hit_rate(base["replies"], NEKO_MARKERS),
        "persona_hit_rate": style_hit_rate(personad["replies"], NEKO_MARKERS),
        "baseline_prompts_tokens": sum(
            len(engine.tok.encode("User: " + p + "\n\nAssistant:")) for p in style_prompts
        ),
        "samples": [
            {"prompt": p, "none": b, "persona": x}
            for p, b, x in zip(style_prompts, base["replies"], personad["replies"])
        ],
    }
    result["serving"] = bench_serving(engine, args.persona, "none", turns=args.turns)
    result["stateless_prefill"] = bench_prefill_only(
        engine, "none", [512, 1024, 2048, 4096]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "style"}, ensure_ascii=False, indent=2))
    s = result["style"]
    print(f"style hit rate: baseline {s['baseline_hit_rate']:.0%} -> persona {s['persona_hit_rate']:.0%}")


if __name__ == "__main__":
    main()
