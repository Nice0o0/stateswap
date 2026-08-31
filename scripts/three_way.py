# Three-way comparison: S0 vs LoRA vs system-prompt (not part of package).
# 三个阵营都在 1.5B 底座上注入同一个猫娘人格（neko 语料），统一贪心评测：
#   风格命中率 / 通用能力保持 / 多人格服务成本
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.engine import Engine  # noqa: E402

MODEL = "models/rwkv7-1.5b-world-hf"
LORA_MODEL = "models/rwkv7-1.5b-world-neko-lora"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"
MARKERS = ("喵", "主人", "本喵", "小鱼干", "（", "(")
STYLE_PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "给我讲个笑话吧",
    "我有点想你了",
    "晚饭吃点什么好呢？",
]
GENERAL = ["What is the capital of France?", "1 加 1 等于几？"]
# system-prompt 阵营：从语料里抽 12 条风格示例
import random  # noqa: E402

corpus = json.loads(Path("data/neko_corpus_full.json").read_text(encoding="utf-8"))
demos = random.Random(7).sample(corpus, 12)
SYS_DEMO = "你必须始终使用以下示例中的说话方式回答（猫娘口吻、动作括号、喵语气词）：\n\n" + "\n\n".join(
    f"User: {d['instruction']}\nAssistant: {d['output']}" for d in demos)
print(f"system-prompt arm: {len(SYS_DEMO)} chars (~{len(SYS_DEMO)//1} chars)")


def style_rate(chat_fn):
    hits, replies = 0, []
    for p in STYLE_PROMPTS:
        r = chat_fn(p)
        replies.append(r)
        if any(m in r for m in MARKERS):
            hits += 1
    return hits / len(STYLE_PROMPTS), replies


out = {}

# ---- Arm 1: S0 ----
engine_s0 = Engine(MODEL, VOCAB)
engine_s0.register_persona("neko-1.5b", "personas/neko-1.5b/s0.pt")
t0 = time.perf_counter()
engine_s0.register_persona("switch-test", "personas/zh2en-1.5b/s0.pt")
switch_cost = (time.perf_counter() - t0) * 1000
s = engine_s0.new_session("neko-1.5b")
rate, replies = style_rate(lambda p: engine_s0.chat(s.session_id, p, max_new_tokens=96,
                                                    temperature=0.0, top_p=1.0)["reply"])
gen = [engine_s0.chat(s.session_id, g, max_new_tokens=48, temperature=0.0, top_p=1.0)["reply"]
       for g in GENERAL]
out["S0"] = {"style": rate, "switch_cost_ms": switch_cost,
             "state_mb": 12.78, "per_request_overhead_tokens": 0,
             "general": dict(zip(GENERAL, [g[:60] for g in gen])), "replies": replies}

# ---- Arm 2: LoRA (merged weights = separate model dir) ----
if Path(LORA_MODEL).exists():
    engine_lora = Engine(LORA_MODEL, VOCAB)
    s = engine_lora.new_session("none")
    rate, replies = style_rate(lambda p: engine_lora.chat(s.session_id, p, max_new_tokens=96,
                                                          temperature=0.0, top_p=1.0)["reply"])
    gen = [engine_lora.chat(s.session_id, g, max_new_tokens=48, temperature=0.0, top_p=1.0)["reply"]
           for g in GENERAL]
    out["LoRA"] = {"style": rate,
                   "switch_cost_ms": "N/A (merge = 换整个模型目录)",
                   "state_mb": "N/A (权重已合并)",
                   "per_request_overhead_tokens": 0,
                   "general": dict(zip(GENERAL, [g[:60] for g in gen])), "replies": replies}
else:
    print("LoRA model not found, skip")

# ---- Arm 3: system prompt ----
engine_p = Engine(MODEL, VOCAB)
s = engine_p.new_session("none")


def with_sys(p):
    return engine_p.chat(s.session_id, SYS_DEMO + "\n\n" + p, max_new_tokens=96,
                         temperature=0.0, top_p=1.0)["reply"]


rate, replies = style_rate(with_sys)
gen = [with_sys(g)[:60] for g in GENERAL]
out["system_prompt"] = {"style": rate,
                        "per_request_overhead_tokens": len(engine_p.tok.encode(SYS_DEMO)),
                        "switch_cost_ms": 0.0,
                        "general": dict(zip(GENERAL, gen)), "replies": replies}

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("replies", "general")}
                  for k, v in out.items()}, ensure_ascii=False, indent=1))
Path("docs").mkdir(exist_ok=True)
Path("docs/three_way.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
print("saved docs/three_way.json")
