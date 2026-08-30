# Validate a converted RWKV-7 model end to end:
#   1. greedy generation from S0=0 is sane text
#   2. gradients reach S0 through the chunk kernel (finite, non-zero)
#   3. one loss step on a NekoQA sample decreases loss
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.data import build_example  # noqa: E402
from stateswap.s0 import S0, load_base_model, make_cache  # noqa: E402
from stateswap.tokenizer import WorldTokenizer  # noqa: E402

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/rwkv7-0.1b-world-hf"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"

torch.manual_seed(0)
tok = WorldTokenizer(VOCAB)
print(f"vocab entries: {len(tok.id_to_bytes)}")
print("model dir:", MODEL_DIR)

model = load_base_model(MODEL_DIR)
print(
    "layers:",
    model.config.num_hidden_layers,
    "heads:",
    model.config.hidden_size // model.config.head_dim,
    "hidden:",
    model.config.hidden_size,
)

# --- 1. generation sanity (S0 = 0) ---
s0 = S0(model).to("cuda")
cache = make_cache(model, s0, batch_size=1, detach_states=True)
ids = tok.encode("User: What is the capital of France?\n\nAssistant:")
with torch.no_grad():
    out = model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache, use_cache=True)
    print("prefill logits:", tuple(out.logits.shape), out.logits.dtype)
    generated = []
    for _ in range(40):
        nxt = int(out.logits[0, -1].argmax())
        generated.append(nxt)
        if tok.decode(generated).rstrip().endswith(("\n\n",)) and len(generated) > 4:
            break
        out = model(
            input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True
        )
print("GEN:", repr("User: What is the capital of France?\n\nAssistant:" + tok.decode(generated))[:400])

# --- 2. gradient check through the real model ---
# model.train() 强制走 chunk 内核路径（fla 层依据 self.training 选内核）；
# RWKV7 无 dropout/BN，train 模式无副作用。
s0_t = S0(model).to("cuda")
s0_t.init_noise(std=0.0)
ex = build_example(
    tok,
    "早上好呀，宝宝！今天想吃小鱼干吗？昨晚梦到你在追蝴蝶呢，说说看嘛？",
    "（耳朵唰地竖起来）喵？！主人怎么知道宝宝梦见蝴蝶的说！（尾巴兴奋地打转）蝴蝶闪闪的翅膀超好看喵！宝宝追了它一整晚，结果...结果撞在了梦里的玻璃上呜呜...（揉脑袋）不过没关系啦！主人要做宝宝的玻璃保护队喵！（扑过来蹭蹭）",
)
input_ids = torch.tensor([ex.input_ids], device="cuda")
labels = torch.tensor([ex.labels], device="cuda")
print("example tokens:", len(ex.input_ids), "prompt_len:", ex.prompt_len)
model.train()
cache = make_cache(model, s0_t, batch_size=1, detach_states=False)
out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
loss = torch.nn.functional.cross_entropy(
    out.logits.float()[:, :-1].reshape(-1, out.logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
)
print("initial loss:", float(loss))
loss.backward()
norms = [float(p.grad.float().norm()) for p in s0_t.states]
print("S0 grad norms: first3 =", [f"{n:.3e}" for n in norms[:3]])
print("all finite:", all(n == n and abs(n) != float("inf") for n in norms))
print("all non-zero:", all(n > 0 for n in norms))

# --- 3. tiny overfit: 10 steps on one sample ---
s0_o = S0(model).to("cuda")
opt = torch.optim.AdamW(s0_o.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0)
model.eval()
for step in range(10):
    cache = make_cache(model, s0_o, batch_size=1, detach_states=False)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    loss = torch.nn.functional.cross_entropy(
        out.logits.float()[:, :-1].reshape(-1, out.logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(s0_o.parameters(), 1.0)
    opt.step()
    if step % 3 == 0 or step == 9:
        print(f"overfit step {step}: loss {float(loss):.4f}")
print("VALIDATION DONE")
