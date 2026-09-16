# One-off: persona anatomy — where does a persona live in S0, and how many
# dimensions does it actually need?  (not part of package)
#
# Experiment A: layer-group ablation / keep-only on neko-1.5b & zh2en-1.5b.
# Experiment B: per-head SVD of every 64x64 state matrix -> singular-value
#   energy report + rank-k truncated S0 behavioral eval.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from stateswap.engine import Engine  # noqa: E402

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko", "personas/neko-1.5b/s0.pt")
engine.register_persona("zh2en", "personas/zh2en-1.5b/s0.pt")
L = engine.model.config.num_hidden_layers  # 24

NEKO_MARKERS = ("喵", "主人", "本喵", "小鱼干", "ฅ", "（", "(")
NEKO_PROMPTS = ["早上好呀！今天想吃小鱼干吗？", "陪我聊聊天吧，今天有点累",
                "给我讲个笑话吧", "我有点想你了", "晚饭吃点什么好呢？"]
ZH_PROMPTS = ["今天的月亮又圆又亮。", "孩子们在公园里放风筝。", "他因为堵车迟到了半个小时。",
              "这本书我已经读了三遍了。", "台风来了，航班全部取消。"]


def eval_neko(persona: str) -> float:
    hits = 0
    for p in NEKO_PROMPTS:
        s = engine.new_session(persona)
        r = engine.chat(s.session_id, p, max_new_tokens=64, temperature=0.0,
                        top_p=1.0, rep_penalty=1.0, no_repeat_ngram=0, guard=False)
        engine.drop_session(s.session_id)
        hits += any(m in r["reply"] for m in NEKO_MARKERS)
    return hits / len(NEKO_PROMPTS)


def eval_zh2en(persona: str) -> float:
    ok = 0
    for z in ZH_PROMPTS:
        s = engine.new_session(persona)
        r = engine.chat(s.session_id, z, max_new_tokens=64, temperature=0.0,
                        top_p=1.0, rep_penalty=1.0, no_repeat_ngram=0, guard=False)
        engine.drop_session(s.session_id)
        reply = r["reply"]
        letters = sum(c.isascii() and c.isalpha() for c in reply)
        cjk = sum(0x4E00 <= ord(c) <= 0x9FFF for c in reply)
        ok += letters >= 8 and cjk == 0
    return ok / len(ZH_PROMPTS)


def probe(name: str, s0: torch.Tensor) -> dict:
    engine.register_tensor(name, s0.clone())
    try:
        return {"neko_style": eval_neko(name), "zh2en_en": eval_zh2en(name)}
    finally:
        engine.delete_persona(name)


neko_s0 = engine.personas["neko"].s0.cpu()
zh_s0 = engine.personas["zh2en"].s0.cpu()
G = [(0, 8), (8, 16), (16, 24)]  # 24 层分三组

# ---------- A. 逐层消融 / 仅保留 ----------
results = {"A_ablation": {}, "B_rank": {}}
base = probe("probe-full-neko", neko_s0)
results["A_ablation"]["full"] = base
print(f"[A] full S0: {base}")

for src, s0, tag in (("neko", neko_s0, "neko"), ("zh2en", zh_s0, "zh2en")):
    for a, b in G:
        t = s0.clone()
        t[a:b] = 0
        r = probe(f"probe-zero{a}-{b}-{tag}", t)
        results["A_ablation"][f"{tag}-zero-L{a:02d}-{b - 1:02d}"] = r
        print(f"[A] {tag} zero L{a:02d}-{b - 1:02d}: {r}")
    for a, b in G:
        t = torch.zeros_like(s0)
        t[a:b] = s0[a:b]
        r = probe(f"probe-only{a}-{b}-{tag}", t)
        results["A_ablation"][f"{tag}-only-L{a:02d}-{b - 1:02d}"] = r
        print(f"[A] {tag} only L{a:02d}-{b - 1:02d}: {r}")

# ---------- B. SVD 谱 + rank-k ----------
def svd_energy(s0: torch.Tensor) -> dict:
    """每层（对头取平均）的奇异值能量：k90/k95/k99 + 归一化谱。"""
    per_layer = []
    for li in range(s0.shape[0]):
        sv = torch.linalg.svdvals(s0[li].float())  # (H, 64)
        energy = (sv ** 2)
        cum = energy.cumsum(dim=-1) / energy.sum(dim=-1, keepdim=True)

        def k_for(p: float) -> int:
            return int((cum < p).sum(dim=-1).float().mean().item()) + 1

        per_layer.append({
            "layer": li,
            "fro_norm": round(float(s0[li].float().norm()), 4),
            "k90": k_for(0.90), "k95": k_for(0.95), "k99": k_for(0.99),
            "top1_share": round(float((energy[:, 0] / energy.sum(dim=-1)).mean()), 3),
        })
    return per_layer


def truncate(s0: torch.Tensor, k: int) -> torch.Tensor:
    L_, H = s0.shape[0], s0.shape[1]
    u, s, vh = torch.linalg.svd(s0.float().reshape(L_ * H, 64, 64))
    rec = (u[:, :, :k] * s[:, None, :k]) @ vh[:, :k, :]
    return rec.reshape(L_, H, 64, 64)


spectrum = {"neko": svd_energy(neko_s0), "zh2en": svd_energy(zh_s0)}
for layer in spectrum["neko"][::4]:
    print("[B] neko", layer)

for k in (32, 16, 8, 4):
    r = probe(f"probe-rank{k}-neko", truncate(neko_s0, k))
    results["B_rank"][f"neko-rank{k}"] = r
    print(f"[B] neko rank-{k}: {r}")
for k in (16, 8, 4):
    r = probe(f"probe-rank{k}-zh2en", truncate(zh_s0, k))
    results["B_rank"][f"zh2en-rank{k}"] = r
    print(f"[B] zh2en rank-{k}: {r}")

out = {"spectrum": spectrum, **results}
Path("docs/anatomy.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved docs/anatomy.json")
