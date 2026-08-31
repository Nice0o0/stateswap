"""StateBench: standardized evaluation for RWKV-7 state tuning.

把项目中验证过的评测方法产品化，输出统一的 scorecard（JSON）：
  1. StyleAdherence      —— 人格风格命中率（标记词法，贪心解码）
  2. KnowledgeInjection  —— 知识注入：记忆保持 / 泛化（合成事实 QA）
  3. GeneralCapability   —— 能力保持（S₀ 注入后底座通用行为是否受损）

设计约定（来自实验教训，见 docs/state-arithmetic.md）：
  - 贪心解码（温度 0），排除采样运气
  - 每个评测探针使用独立会话（会话状态携带历史，混用即污染）
  - 知识注入的评测集需要"训练内实体 + 换问法"与"未训实体"两组

用法：
    python -m stateswap.benchsuite --model models/rwkv7-1.5b-world-hf \
        --persona neko-1.5b --knowledge data/knowledge_eval.json \
        --style-markers 喵,主人,小鱼干 --out scorecard.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .engine import Engine

_NORM = re.compile(r"[\s，。？！,.?!、；;：:'\"“”‘’（）()]")


def norm(s: str) -> str:
    return _NORM.sub("", s)


def contains_answer(reply: str, gold: str) -> bool:
    g, r = norm(gold), norm(reply)
    return bool(g) and g in r


@dataclass
class StyleConfig:
    markers: tuple[str, ...] = ("喵", "主人", "本喵")
    prompts: tuple[str, ...] = (
        "早上好呀！今天想吃小鱼干吗？",
        "陪我聊聊天吧，今天有点累",
        "给我讲个笑话吧",
        "我有点想你了",
        "晚饭吃点什么好呢？",
    )


@dataclass
class GeneralProbeConfig:
    prompts: tuple[str, ...] = (
        "What is the capital of France?",
        "1 加 1 等于几？",
        "用一句话介绍你自己。",
    )


@dataclass
class Scorecard:
    persona: str
    style: dict = field(default_factory=dict)
    knowledge: dict = field(default_factory=dict)
    general: dict = field(default_factory=dict)


class StateBench:
    def __init__(self, engine: Engine):
        self.engine = engine

    def _ask(self, persona: str, question: str, max_new_tokens: int = 64) -> str:
        """每个探针独立会话：会话状态携带对话历史，混用即污染。"""
        session = self.engine.new_session(persona)
        try:
            return self.engine.chat(
                session.session_id, question,
                max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0,
            )["reply"].strip()
        finally:
            self.engine.drop_session(session.session_id)

    # ---- 1. style adherence ----

    def style_adherence(self, persona: str, config: StyleConfig) -> dict:
        replies = []
        hits = 0
        for p in config.prompts:
            r = self._ask(persona, p, max_new_tokens=96)
            replies.append(r)
            if any(m in r for m in config.markers):
                hits += 1
        return {"hit_rate": hits / max(1, len(config.prompts)),
                "prompts": list(config.prompts), "replies": replies}

    # ---- 2. knowledge injection ----

    def knowledge_injection(self, persona: str, eval_file: str | Path) -> dict:
        data = json.loads(Path(eval_file).read_text(encoding="utf-8"))
        out = {}
        for split in ("memorization", "generalization"):
            rows = []
            for q in data.get(split, []):
                reply = self._ask(persona, q["instruction"])
                rows.append({"q": q["instruction"], "gold": q["output"],
                             "reply": reply, "correct": contains_answer(reply, q["output"])})
            out[split] = {
                "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
                "detail": rows,
            }
        out["note"] = ("memorization=训练内实体换问法；generalization=未训实体，"
                       "衡量 schema 泛化（S₀ 注入通常接近 0，如实报告）")
        return out

    # ---- 3. general capability (forgetting probe) ----

    def general_capability(self, persona: str, config: GeneralProbeConfig) -> dict:
        replies = []
        for q in config.prompts:
            replies.append({"q": q, "reply": self._ask(persona, q, max_new_tokens=96)})
        return {"probes": replies,
                "note": "定性对照：与基线人格的回复并排人工比较，检查通用能力是否被 S₀ 破坏"}

    # ---- scorecard ----

    def run_scorecard(
        self,
        persona: str,
        style: StyleConfig | None = None,
        knowledge_file: str | Path | None = None,
        general: GeneralProbeConfig | None = None,
    ) -> Scorecard:
        card = Scorecard(persona=persona)
        card.style = self.style_adherence(persona, style or StyleConfig())
        if knowledge_file:
            card.knowledge = self.knowledge_injection(persona, knowledge_file)
        card.general = self.general_capability(persona, general or GeneralProbeConfig())
        return card


def main() -> None:
    ap = argparse.ArgumentParser(description="StateBench scorecard")
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")
    ap.add_argument("--persona", required=True)
    ap.add_argument("--knowledge", default=None, help="knowledge eval json (data/knowledge_eval.json)")
    ap.add_argument("--style-markers", default="喵,主人,本喵")
    ap.add_argument("--out", default="docs/scorecard.json")
    args = ap.parse_args()

    engine = Engine(args.model, args.vocab)
    if args.persona not in engine.personas:
        engine.register_persona(args.persona, f"personas/{args.persona}/s0.pt")

    bench = StateBench(engine)
    style_cfg = StyleConfig(markers=tuple(m.strip() for m in args.style_markers.split(",") if m.strip()))
    card = bench.run_scorecard(args.persona, style=style_cfg, knowledge_file=args.knowledge)

    payload = {
        "persona": card.persona,
        "style": {k: v for k, v in card.style.items() if k != "replies"},
        "knowledge": {k: ({"accuracy": v["accuracy"]} if isinstance(v, dict) and "accuracy" in v else v)
                      for k, v in card.knowledge.items()},
        "general": card.general,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "general"}, ensure_ascii=False, indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
