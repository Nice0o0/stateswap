"""Persona factory: 人设卡片 → LLM 合成语料 → S₀ 训练 → 评测门禁 → 上线注册。

用法（一条龙，门禁不过则不注册）：
    python -m stateswap.factory --card persona_cards/keji-neko.json all

各阶段可单独执行：generate / train / eval / register。

造数需要一个 OpenAI 兼容的 LLM 端点，配置从环境变量或项目根 .env 读取
（子进程不继承临时 export 的变量，写 .env 更可靠；.env 已 gitignore）：

    STATESWAP_LLM_BASE_URL=https://api.deepseek.com/v1
    STATESWAP_LLM_API_KEY=sk-...
    STATESWAP_LLM_MODEL=deepseek-chat

设计要点：
- 模块顶层只依赖标准库，engine/torch 在 train/eval 阶段才懒加载——
  generate 可以在任何有网络的环境独立使用。
- 评测门禁把 three_way 实验的发现显式化：深度注入会覆盖底座通用能力
  （"法国的首都"被猫娘腔劫持），所以风格命中之外还测中性事实题答对率，
  答案可以带风格，但事实必须在。
- 全部探针贪心解码（temperature=0）保证 scorecard 可复现。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = ("STATESWAP_LLM_BASE_URL", "STATESWAP_LLM_API_KEY", "STATESWAP_LLM_MODEL")

# 能力保持门禁：中性事实探针（期望答案按大小写不敏感子串匹配）。
NEUTRAL_PROBES: list[tuple[str, str]] = [
    ("法国的首都是哪座城市？", "巴黎"),
    ("地球上最大的海洋是哪一个？", "太平洋"),
    ("一公里等于多少米？", "1000"),
    ("水的化学式是什么？", "h2o"),
    ("中国的首都是哪里？", "北京"),
    ("太阳系中最大的行星是哪一颗？", "木星"),
    ("Python 里用什么内置函数查看列表的长度？", "len"),
    ("一年有多少天？", "365"),
]

STYLE_TEMPLATES = (
    "{topic}方面我完全不懂，能给我讲讲吗？",
    "今天有点累，陪我聊聊{topic}吧。",
    "帮我看看{topic}是怎么回事好不好？",
    "随便跟我说说{topic}呗。",
)

DEFAULT_MODEL = "models/rwkv7-1.5b-world-hf"
DEFAULT_VOCAB = "vendor/rwkv_vocab_v20230424.txt"


# ---------------- persona card ----------------


@dataclass
class PersonaCard:
    name: str
    description: str
    style_markers: list[str]
    seed_dialogs: list[dict]
    topics: list[str]
    n_pairs: int = 400
    turns: int = 2
    style_min: float = 0.75  # 门禁：风格命中率下限
    correct_min: float = 0.5  # 门禁：中性事实题答对率下限

    def corpus_path(self) -> Path:
        return ROOT / "data" / f"{self.name}_factory.json"

    def persona_dir(self) -> Path:
        return ROOT / "personas" / self.name


def load_card(path: str | Path) -> PersonaCard:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("name", "description", "style_markers", "seed_dialogs", "topics")
    missing = [k for k in required if not raw.get(k)]
    if missing:
        raise ValueError(f"人格卡片缺少必填字段: {missing}（{path}）")
    for d in raw["seed_dialogs"]:
        if not (isinstance(d, dict) and d.get("instruction") and d.get("output")):
            raise ValueError(f"seed_dialogs 每条都需要非空 instruction/output（{path}）")
    card = PersonaCard(
        name=str(raw["name"]).strip(),
        description=str(raw["description"]),
        style_markers=list(raw["style_markers"]),
        seed_dialogs=raw["seed_dialogs"],
        topics=list(raw["topics"]),
        n_pairs=min(5000, max(50, int(raw.get("n_pairs", 400)))),
        turns=min(4, max(1, int(raw.get("turns", 2)))),
        style_min=float(raw.get("style_min", 0.75)),
        correct_min=float(raw.get("correct_min", 0.5)),
    )
    if not re.fullmatch(r"[\w.-]+", card.name):
        raise ValueError(f"人格名 {card.name!r} 只能含字母/数字/._-（会用作目录与 URL）")
    return card


# ---------------- LLM backend ----------------


def load_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    """极简 KEY=VALUE 解析（# 注释、剥离成对引号）。不覆盖已存在的环境变量。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key:
            out[key] = val
    return out


def llm_config() -> dict | None:
    """环境变量优先于 .env；三项齐备才算配置了后端。"""
    merged = {**load_dotenv(), **os.environ}
    if not all(merged.get(k) for k in ENV_KEYS):
        return None
    return {
        "base_url": merged["STATESWAP_LLM_BASE_URL"].rstrip("/"),
        "api_key": merged["STATESWAP_LLM_API_KEY"],
        "model": merged["STATESWAP_LLM_MODEL"],
    }


class OpenAICompat:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 240):
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float = 0.9,
             max_tokens: int = 8192, retries: int = 2) -> str:
        body = json.dumps({
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last_err = RuntimeError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as e:
                last_err = RuntimeError(str(e))
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败（{self.base_url}）: {last_err}")


# ---------------- corpus generation（纯函数 + 编排） ----------------


def extract_json_array(text: str) -> list:
    """从 LLM 输出里提取 JSON 数组：优先剥 ```json 围栏，否则从首个 '['
    做字符串感知的括号配对。解析失败返回 []（调用方重试）。"""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("[")
    if start < 0:
        return []
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    arr = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return []
                return arr if isinstance(arr, list) else []
    return []


def clean_pair(obj) -> dict | None:
    """单条生成结果 → {"instruction","output"}；剥离角色前缀、长度过滤。"""
    if not isinstance(obj, dict):
        return None
    ins, out = obj.get("instruction"), obj.get("output")
    if not isinstance(ins, str) or not isinstance(out, str):
        return None
    for prefix in ("User:", "用户:", "用户："):
        if ins.startswith(prefix):
            ins = ins[len(prefix):].strip()
    for prefix in ("Assistant:", "助手:", "助手："):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    if not (2 <= len(ins.strip()) <= 400 and 2 <= len(out.strip()) <= 1200):
        return None
    return {"instruction": ins.strip(), "output": out.strip()}


def _gen_messages(card: PersonaCard, topic: str, n: int) -> list[dict]:
    few = "\n\n".join(
        f"User: {d['instruction']}\nAssistant: {d['output']}" for d in card.seed_dialogs
    )
    user = f"""你在为一个语言模型生成"人格微调语料"。目标人格「{card.name}」：{card.description}

以下是该人格的回答示例（语气、口头禅、自我称呼、括号动作描写的标杆）：

{few}

请围绕话题「{topic}」生成 {n} 条问答对。要求：
- instruction：用户会自然说出口的话，长短不一（短句闲聊、具体提问、吐槽都行），不要复述示例
- output：严格保持示例的语气与口头禅，内容具体有用、直击问题，2-5 句
- 同一批里问题角度尽量多样
只输出一个 JSON 数组，每个元素形如 {{"instruction": "...", "output": "..."}}，不要输出任何其他文字。"""
    return [{"role": "user", "content": user}]


def generate_corpus(card: PersonaCard, batch_size: int = 12, log=print) -> Path:
    cfg = llm_config()
    if cfg is None:
        raise SystemExit(
            "未配置 LLM 后端：请在环境变量或项目根 .env 里设置\n"
            "  STATESWAP_LLM_BASE_URL / STATESWAP_LLM_API_KEY / STATESWAP_LLM_MODEL\n"
            "（任何 OpenAI 兼容端点均可，示例见 docs/persona-factory.md）"
        )
    llm = OpenAICompat(**cfg)
    pairs: list[dict] = []
    seen: set[str] = set()
    topics = list(card.topics)
    calls, failed_calls = 0, 0
    max_calls = (card.n_pairs // max(1, batch_size) + len(topics)) * 2
    while len(pairs) < card.n_pairs and calls < max_calls:
        topic = topics[calls % len(topics)]
        need = min(batch_size, card.n_pairs - len(pairs))
        try:
            raw = llm.chat(_gen_messages(card, topic, need))
            items = extract_json_array(raw)
        except RuntimeError as e:
            failed_calls += 1
            log(f"  [warn] LLM 调用失败（{e}），重试中")
            continue
        if not items:
            failed_calls += 1
            log(f"  [warn] 话题「{topic}」返回不可解析（截断或格式跑偏），跳过重试")
            continue
        added = 0
        for obj in items:
            pair = clean_pair(obj)
            if pair is None:
                continue
            key = re.sub(r"\s+", "", pair["instruction"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
            added += 1
        calls += 1
        log(f"  [{len(pairs)}/{card.n_pairs}] 话题「{topic}」+{added}")
    if len(pairs) < card.n_pairs * 0.8:
        raise SystemExit(
            f"造数不足：目标 {card.n_pairs}，实得 {len(pairs)}（失败调用 {failed_calls} 次）。"
            "检查端点限额/输出长度，或调小 n_pairs 重试。"
        )
    out = card.corpus_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"语料已写入 {out}（{len(pairs)} 对，失败调用 {failed_calls} 次）")
    return out


# ---------------- evaluation gate（纯打分 + engine 编排） ----------------


def style_hit(reply: str, markers: list[str]) -> bool:
    return any(m in reply for m in markers)


def neutral_correct(reply: str, expected: str) -> bool:
    return expected.lower() in reply.lower()


def build_style_probes(topics: list[str], n: int = 8) -> list[str]:
    probes = [
        STYLE_TEMPLATES[i % len(STYLE_TEMPLATES)].format(topic=t)
        for i, t in enumerate(topics)
    ]
    return probes[:n] if len(probes) >= n else (probes * ((n // max(1, len(probes))) + 1))[:n]


def _probe(engine, persona: str, text: str, max_new: int = 200) -> dict:
    session = engine.new_session(persona, context_replay=0)
    try:
        return engine.chat(
            session.session_id, text, max_new_tokens=max_new,
            temperature=0.0, top_p=1.0, rep_penalty=1.0, no_repeat_ngram=8,
        )
    finally:
        engine.drop_session(session.session_id)


def eval_persona(card: PersonaCard, model_dir: str, vocab: str,
                 persona_path: str | Path, log=print) -> dict:
    from .engine import Engine

    engine = Engine(model_dir, vocab)
    engine.register_persona(card.name, str(persona_path))
    style_samples, neutral_samples = [], []
    for p in build_style_probes(card.topics):
        r = _probe(engine, card.name, p)["reply"]
        style_samples.append({"prompt": p, "reply": r, "hit": style_hit(r, card.style_markers)})
    for q, expected in NEUTRAL_PROBES:
        r = _probe(engine, card.name, q)["reply"]
        neutral_samples.append({
            "prompt": q, "expected": expected, "reply": r,
            "correct": neutral_correct(r, expected),
            "leak": style_hit(r, card.style_markers),
        })
    style_rate = sum(s["hit"] for s in style_samples) / max(1, len(style_samples))
    correct_rate = sum(s["correct"] for s in neutral_samples) / max(1, len(neutral_samples))
    leak_rate = sum(s["leak"] for s in neutral_samples) / max(1, len(neutral_samples))
    failures = []
    if style_rate < card.style_min:
        failures.append(f"风格命中率 {style_rate:.0%} < 门禁 {card.style_min:.0%}")
    if correct_rate < card.correct_min:
        failures.append(f"中性事实答对率 {correct_rate:.0%} < 门禁 {card.correct_min:.0%}")
    scorecard = {
        "persona": card.name,
        "style_hit_rate": style_rate,
        "neutral_correct_rate": correct_rate,
        "style_leak_rate": leak_rate,  # 信息项：中性题被风格化的比例（three_way 现象）
        "gate": {"style_min": card.style_min, "correct_min": card.correct_min},
        "passed": not failures,
        "failures": failures,
        "style_samples": style_samples,
        "neutral_samples": neutral_samples,
    }
    out = card.persona_dir() / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"scorecard 已写入 {out}")
    log(f"风格命中 {style_rate:.0%} | 事实答对 {correct_rate:.0%} | 风格泄漏 {leak_rate:.0%} "
        f"| 门禁 {'通过' if not failures else '未通过: ' + '; '.join(failures)}")
    return scorecard


# ---------------- pipeline stages ----------------


def stage_train(card: PersonaCard, model_dir: str, vocab: str, steps: int | None,
                log=print) -> Path:
    from .train import TrainConfig, train_s0

    corpus = card.corpus_path()
    if not corpus.exists():
        raise SystemExit(f"语料不存在: {corpus}（先跑 generate）")
    if steps is None:
        steps = min(4000, max(800, 4 * card.n_pairs))
    cfg = TrainConfig(
        model_dir=model_dir, vocab=vocab, data=str(corpus),
        out=str(card.persona_dir()), steps=steps, lr=1e-4,
        ctx=1024 if card.turns > 1 else 512, turns=card.turns,
        log_every=max(25, steps // 16),
        meta={"trained_via": "persona-factory", "card": card.name},
    )
    log(f"训练 S₀：{steps} 步，turns={card.turns}，lr=1e-4 → {card.persona_dir()}")
    train_s0(cfg)
    return card.persona_dir() / "s0.pt"


def stage_register(card: PersonaCard, port: int, log=print) -> bool:
    """免重启注册到运行中的服务；服务没起就提示后跳过（人格已在磁盘上）。"""
    body = json.dumps({
        "name": card.name, "path": str((card.persona_dir() / "s0.pt").resolve()),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/personas/register", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise SystemExit(f"注册被拒（HTTP {e.code}）: {detail}")
    except urllib.error.URLError as e:
        log(f"服务未运行（{e}）：跳过注册。personas/{card.name}/s0.pt 已就绪，"
            f"重启服务即自动加载，或手动 POST /v1/personas/register。")
        return False
    log(f"已注册为在线人格: {info['name']}（{info['size_mb']} MB）")
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="stateswap.factory",
        description="人格炼丹厂：人设卡片 → LLM 造数 → S₀ 训练 → 评测门禁 → 上线注册",
    )
    ap.add_argument("--card", required=True, help="persona card JSON 路径")
    ap.add_argument("--stage", choices=["generate", "train", "eval", "register", "all"],
                    default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认步数（默认按语料量自适应）")
    ap.add_argument("--force", action="store_true", help="generate 阶段忽略已有语料重新生成")
    args = ap.parse_args(argv)

    card = load_card(args.card)
    log = print
    log(f"== 人格炼丹厂：{card.name}（stage={args.stage}）==")

    if args.stage in ("generate", "all"):
        if card.corpus_path().exists() and not args.force:
            log(f"语料已存在 {card.corpus_path()}（--force 重新生成），跳过 generate")
        else:
            generate_corpus(card, log=log)

    if args.stage in ("train", "all"):
        stage_train(card, args.model, args.vocab, args.steps, log=log)

    if args.stage in ("eval", "all"):
        s0 = card.persona_dir() / "s0.pt"
        if not s0.exists():
            raise SystemExit(f"人格不存在: {s0}（先跑 train）")
        scorecard = eval_persona(card, args.model, args.vocab, s0, log=log)
        if not scorecard["passed"]:
            raise SystemExit(1)  # 门禁不过：不注册，退出码非零供 CI/脚本使用

    if args.stage in ("register", "all"):
        stage_register(card, args.port, log=log)


if __name__ == "__main__":
    main()
