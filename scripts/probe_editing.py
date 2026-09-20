# One-off: S₀ 低秩定向编辑实验（E1 风格切除 / E2 任务切除 / E3 粗编辑基线 /
# E4 定向注入 / E5 几何 / E6 退化记录）。不是包的一部分。
#
# 对象：mix-0.60（共存相：风格 87.5% + 任务 75%，见 docs/phase-transition.md）。
# 协议与相图实验一致：贪心、独立会话、context_replay=0、8+8 探针。
# 结果增量写入 docs/state-editing.json；报告 docs/state-editing.md。
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from probe_phase_transition import ROOT, Probe, save  # noqa: E402

from stateswap.arithmetic import subtract  # noqa: E402
from stateswap.editing import inject, remove_subspace  # noqa: E402

OUT = ROOT / "docs" / "state-editing.json"
BASE_ALPHA = 0.60  # 共存相
RANKS = [1, 2, 3, 4, 8, 16]


def thirds_cos(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    cs = []
    for i in range(a.shape[0]):
        x = a[i].reshape(a.shape[1], -1).float()
        y = b[i].reshape(b.shape[1], -1).float()
        cs.append(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean().item())
    n = a.shape[0] // 3
    tail = a.shape[0] - 2 * n
    return [round(sum(cs[:n]) / n, 4), round(sum(cs[n:2 * n]) / n, 4),
            round(sum(cs[2 * n:]) / tail, 4)]


def geometry(p: Probe, tensor: torch.Tensor, base: torch.Tensor) -> dict:
    return {
        "fro_dist_base": round(float((tensor - base).norm()), 4),
        "cos_base_thirds": thirds_cos(tensor, base),
        "cos_neko_thirds": thirds_cos(tensor, p.neko),
        "cos_zh2en_thirds": thirds_cos(tensor, p.zh2en),
    }


def measure(p: Probe, name: str, tensor: torch.Tensor, base: torch.Tensor,
            results: dict, tag: str) -> dict:
    p.engine.register_tensor(name, tensor, {"edit": tag})
    style = p.style_probe(persona=name)
    task = p.task_probe(persona=name)
    row = {
        "tag": tag,
        "style_hit_rate": p.style_rate(style),
        "task_english_rate": p.task_rate(task),
        "degenerated": sum(s["degenerated"] for s in style)
                       + sum(t["degenerated"] for t in task),
        "style_sample": style[0]["reply"][:100],
        "geometry": geometry(p, tensor, base),
        "style": style, "task": task,
    }
    results["configs"].append(row)
    save(results, OUT)
    p.engine.delete_persona(name)
    print(f"{tag}: style={row['style_hit_rate']:.2f} task={row['task_english_rate']:.2f} "
          f"degen={row['degenerated']}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="仅基线 + E1 k=4")
    args = ap.parse_args()

    p = Probe(str(ROOT / "models" / "rwkv7-1.5b-world-hf"),
              str(ROOT / "vendor" / "rwkv_vocab_v20230424.txt"))
    results = {
        "meta": {"base_alpha": BASE_ALPHA, "ranks": RANKS,
                 "protocol": "greedy, fresh sessions, context_replay=0"},
        "configs": [],
    }
    save(results, OUT)
    base = p.mix_tensor(BASE_ALPHA)
    base_name = p.register_mix(BASE_ALPHA)

    # 基线：未编辑的共存相混合体
    measure(p, base_name, base, base, results, f"base-mix-{BASE_ALPHA}")

    if args.smoke:
        measure(p, "edit-smoke", remove_subspace(base, p.neko, 4), base, results,
                "e1-remove-style-k4")
        return

    # E1 风格切除（ref=neko 的 top-k 奇异子空间）
    for k in RANKS:
        measure(p, f"edit-e1-{k:02d}", remove_subspace(base, p.neko, k), base,
                results, f"e1-remove-style-k{k}")
    # E2 任务切除（ref=zh2en，对偶）
    for k in RANKS:
        measure(p, f"edit-e2-{k:02d}", remove_subspace(base, p.zh2en, k), base,
                results, f"e2-remove-task-k{k}")
    # E3 粗编辑基线（标量算术）
    measure(p, "edit-e3-sub", subtract(base, p.zh2en), base, results,
            "e3-subtract-zh2en")
    for lam in (0.3, 0.6):
        measure(p, f"edit-e3-s{lam}", base - lam * p.neko, base, results,
                f"e3-scale-neko-{lam}")
    # E4 定向注入（沿风格方向推回应有相）
    for lam in (0.2, 0.4):
        measure(p, f"edit-e4-{lam}", inject(base, p.neko, lam, 4), base, results,
                f"e4-inject-neko-{lam}-k4")
    print("done")


if __name__ == "__main__":
    main()
