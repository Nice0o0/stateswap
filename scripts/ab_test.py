# A/B behavioral test: same prompts, S0=0 (baseline) vs trained persona S0.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.s0 import S0, load_base_model, make_cache  # noqa: E402
from stateswap.tokenizer import WorldTokenizer  # noqa: E402

MODEL_DIR = sys.argv[1]
S0_PATH = sys.argv[2] if len(sys.argv) > 2 else None

PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "What is the capital of France?",
    "陪我聊聊天吧，今天有点累",
]

torch.manual_seed(7)
tok = WorldTokenizer("vendor/rwkv_vocab_v20230424.txt")
model = load_base_model(MODEL_DIR)
model.eval()


def generate(s0_module: S0 | None, prompt: str, max_new: int = 80) -> str:
    if s0_module is None:
        s0_module = S0(model)
        s0_module.to("cuda")
    cache = make_cache(model, s0_module, batch_size=1, detach_states=True)
    ids = tok.encode("User: " + prompt + "\n\nAssistant:")
    out = model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True)
    gen = []
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax())
        gen.append(nxt)
        text = tok.decode(gen)
        if text.endswith("\n\n") or text.rstrip().endswith(("User:",)):
            break
        out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
    return tok.decode(gen).split("\n\n")[0]


baseline_s0 = S0(model).to("cuda")
persona_s0 = None
if S0_PATH:
    payload = torch.load(S0_PATH, weights_only=False)
    persona_s0 = S0(model).to("cuda")
    persona_s0.load_stacked(payload["s0"])
    print("persona meta:", payload.get("meta"))

print("=" * 72)
for p in PROMPTS:
    print(f"\n[PROMPT] {p}")
    print(f"[S0=0    ] {generate(baseline_s0, p)!r}")
    if persona_s0 is not None:
        print(f"[PERSONA ] {generate(persona_s0, p)!r}")
print("=" * 72)
