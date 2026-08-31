# One-off: measure the saved S0's TRAINING loss directly (not part of package).
# 训练前向 = model.train() + chunk 内核 + 训练样本 + make_cache(detach=False)。
# 若 CE≈0 而推理召回失败 → 矛盾在推理路径构造；若 CE 高 → 保存的 S0 与训练末态不符。
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.s0 import S0, load_base_model, make_cache  # noqa: E402
from stateswap.tokenizer import WorldTokenizer  # noqa: E402

tok = WorldTokenizer("vendor/rwkv_vocab_v20230424.txt")
model = load_base_model("models/rwkv7-1.5b-world-hf")
model.train()
s0 = S0(model).to("cuda")
payload = torch.load("personas/mem-100/s0.pt", weights_only=False)
s0.load_stacked(payload["s0"])
meta = payload.get("meta", {})
print("meta:", {k: meta.get(k) for k in ("model_dir", "data", "steps", "ctx")})

train = json.loads(Path("data/knowledge_qa_100.json").read_text(encoding="utf-8"))[:10]
from stateswap.data import build_example  # noqa: E402

for q in train:
    ex = build_example(tok, q["instruction"], q["output"], ctx=256)
    input_ids = torch.tensor([ex.input_ids], device="cuda")
    labels = torch.tensor([ex.labels], device="cuda")
    cache = make_cache(model, s0, batch_size=1, detach_states=False)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    vocab = out.logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(
        out.logits.float()[:, :-1].reshape(-1, vocab),
        labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    # 训练条件下的贪心续写（与训练 forward 同路径同状态）
    cache2 = make_cache(model, s0, batch_size=1, detach_states=True)
    with torch.no_grad():
        o2 = model(input_ids=input_ids[:, : ex.prompt_len], past_key_values=cache2, use_cache=True)
        gen = []
        for _ in range(16):
            nxt = int(o2.logits[0, -1].argmax())
            gen.append(nxt)
            o2 = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                       past_key_values=cache2, use_cache=True)
    print(f"CE {float(loss):.4f} | train-conditional greedy: {tok.decode(gen)!r:.60} | gold {q['output']!r}")
