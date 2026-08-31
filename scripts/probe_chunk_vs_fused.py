# One-off: chunk-vs-fused prefill comparison for S0 recall (not part of package).
# 若 train()（chunk 路径）召回正常而 eval()（fused 路径）失败 →
# fused_recurrent 前向对 initial_state 的处理也有 bug。
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

_norm = __import__("re").compile(r"[\s，。？！,.?!、；;：:'\"“”‘’（）()]")


def grade(reply, gold):
    g, r = _norm.sub("", gold), _norm.sub("", reply)
    return bool(g) and g in r


engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("mem-100", "personas/mem-100/s0.pt")
train = json.loads(Path("data/knowledge_qa_100.json").read_text(encoding="utf-8"))[:8]


def greedy_reply(prompt, n=24):
    cache = engine._cache_for(engine.personas["mem-100"])
    ids = engine.tok.encode(prompt)
    out = engine.model(input_ids=torch.tensor([ids], device="cuda"),
                       past_key_values=cache, use_cache=True)
    gen = []
    with torch.no_grad():
        for _ in range(n):
            nxt = int(out.logits[0, -1].argmax())
            gen.append(nxt)
            text = engine.tok.decode(gen)
            if text.endswith("\n\n"):
                break
            out = engine.model(input_ids=torch.tensor([[nxt]], device="cuda"),
                               past_key_values=cache, use_cache=True)
    return engine.tok.decode(gen).split("\n\n")[0]


for mode in ("eval (fused prefill)", "train (chunk prefill)"):
    engine.model.train(mode=(mode.startswith("train")))
    print(f"=== {mode} ===")
    ok = 0
    for q in train:
        r = greedy_reply(f"User: {q['instruction']}\n\nAssistant: ")
        good = grade(r, q["output"])
        ok += good
        print(f"  {'✓' if good else '✗'} {q['instruction']} -> {r!r} (gold {q['output']!r})")
    print(f"  accuracy: {ok}/{len(train)}")
engine.model.eval()
