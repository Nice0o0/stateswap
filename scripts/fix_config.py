# One-off: fix stale num_heads in converted model configs (not part of package).
import json
from pathlib import Path

for d in Path("models").glob("rwkv7-*-hf"):
    cfg_path = d / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    correct = cfg["hidden_size"] // cfg["head_dim"]
    print(d.name, "hidden:", cfg["hidden_size"], "num_heads:", cfg.get("num_heads"), "->", correct)
    cfg["num_heads"] = correct
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
