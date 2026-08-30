# One-off runner: convert a BlinkDL pth into fla/HF format.
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stateswap.convert import convert_rwkv7_pth  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="bfloat16")
    args = ap.parse_args()
    out = convert_rwkv7_pth(args.pth, args.out, precision=args.precision)
    print("saved to", out)
