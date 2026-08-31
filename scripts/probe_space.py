# One-off: verify the trailing-space prompt boundary hypothesis (not part of package).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("mem-100", "personas/mem-100/s0.pt")
import json  # noqa: E402

train = json.loads(Path("data/knowledge_qa_100.json").read_text(encoding="utf-8"))[:5]


def greedy_reply(prompt: str, n=24) -> str:
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


for q in train:
    gold = q["output"]
    r_nospace = greedy_reply(f"User: {q['instruction']}\n\nAssistant:")
    r_space = greedy_reply(f"User: {q['instruction']}\n\nAssistant: ")
    print(f"Q: {q['instruction']}")
    print(f"  gold: {gold!r}")
    print(f"  nospace: {r_nospace!r}")
    print(f"  space  : {r_space!r}")
