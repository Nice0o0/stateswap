# One-off: train a LoRA persona with the SAME parameter budget as S0 (not part
# of the package). 1.5B / r=16 on r/k/v/o + ffn key/value ≈ 3.1M trainable
# params = fair comparison against the 3.1M-param S0.
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from stateswap.data import S0Dataset, collate  # noqa: E402
from stateswap.s0 import make_cache  # noqa: E402
from stateswap.s0 import S0 as S0Holder  # noqa: E402
from stateswap.tokenizer import load_tokenizer  # noqa: E402

MODEL = "models/rwkv7-1.5b-world-hf"
VOCAB = "vendor/rwkv_vocab_v20230424.txt"
DATA = "data/neko_corpus_full.json"
STEPS, LR, CTX = 4000, 1e-4, 512

torch.manual_seed(42)
tok = load_tokenizer(VOCAB)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
)
model = get_peft_model(model, lora_cfg)
model.train()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {trainable:,} ({trainable*4/1e6:.2f} MB fp32)")

dataset = S0Dataset(tok, DATA, ctx=CTX)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, betas=(0.9, 0.95), weight_decay=0)

# LoRA 不需要 S0（标准零初始化状态），但复用 make_cache 的零状态构造
zero_holder = S0Holder(model).cuda()

order = list(range(len(dataset)))
cursor, running, running_n = 0, 0.0, 0
t0 = time.time()
for step in range(STEPS):
    lr = LR * (step + 1) / 500 if step < 500 else (
        1e-5 + 0.5 * (LR - 1e-5) * (1 + __import__("math").cos(
            __import__("math").pi * min(1.0, (step - 500) / (STEPS - 500)))))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)

    if cursor >= len(order):
        import random
        random.shuffle(order)
        cursor = 0
    ex = dataset[order[cursor]]
    cursor += 1
    input_ids, labels = collate([ex])
    input_ids = torch.tensor(input_ids, device="cuda")
    labels_t = torch.tensor(labels, device="cuda")
    cache = make_cache(model, zero_holder, batch_size=1, detach_states=True)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    vocab = out.logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(
        out.logits.float()[:, :-1].reshape(-1, vocab),
        labels_t[:, 1:].reshape(-1), ignore_index=-100)
    if not torch.isfinite(loss):
        print(f"[warn] step {step+1}: non-finite loss, skip")
        opt.zero_grad(set_to_none=True)
        continue
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    if not torch.isfinite(gnorm):
        print(f"[warn] step {step+1}: non-finite grad, skip")
        opt.zero_grad(set_to_none=True)
        continue
    opt.step()

    running += float(loss.detach())
    running_n += 1
    if (step + 1) % 500 == 0 or step == 0:
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"step {step+1:>5}/{STEPS}  loss {running/running_n:.4f}  lr {lr:.2e}  "
              f"|g| {float(gnorm):.2e}  {time.time()/(step+1):.2f}s/step  peak {mem:.1f}GB")
        running, running_n = 0.0, 0

print("merging LoRA into base weights...")
merged = model.merge_and_unload()
out_dir = Path("models/rwkv7-1.5b-world-neko-lora")
out_dir.mkdir(parents=True, exist_ok=True)
from safetensors.torch import save_file  # noqa: E402

sd = {k: v.contiguous() for k, v in merged.state_dict().items()}
save_file(sd, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
merged.config.save_pretrained(out_dir)
print("saved", out_dir)
