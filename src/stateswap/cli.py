"""stateswap CLI."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stateswap",
        description="One RWKV-7 base model, many persona states.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("convert", help="BlinkDL pth -> fla/HF format")
    p.add_argument("--pth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--precision", default="bfloat16")

    p = sub.add_parser("train", help="train an S0 persona state")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--turns", type=int, default=1,
                   help=">1: chain pairs into multi-turn samples (long-conversation robustness)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--init-from", default=None,
                   help="warm start: load an existing S0 (persona inheritance)")
    p.add_argument("--train-layers", default=None, metavar="LO:HI",
                   help="only train layers [LO,HI), e.g. 0:8")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="held-out fraction for eval (0 disables)")
    p.add_argument("--eval-every", type=int, default=0,
                   help=">0: evaluate every N steps and keep s0.best.pt")
    p.add_argument("--resume", default=None, help="resume from <dir>/s0.pt")

    p = sub.add_parser("serve", help="run the OpenAI-compatible server")
    p.add_argument("--model", required=True)
    p.add_argument("--persona-dir", default="personas")
    p.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("chat", help="interactive chat with a persona")
    p.add_argument("--model", required=True)
    p.add_argument("--persona", default="none")
    p.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")

    args = parser.parse_args()
    if args.cmd == "convert":
        from .convert import convert_rwkv7_pth

        convert_rwkv7_pth(args.pth, args.out, precision=args.precision)
        print("saved to", args.out)
    elif args.cmd == "train":
        from .train import TrainConfig, train_s0

        train_s0(
            TrainConfig(model_dir=args.model, vocab=args.vocab, data=args.data, out=args.out,
                        steps=args.steps, lr=args.lr, ctx=args.ctx, turns=args.turns,
                        batch_size=args.batch_size, init_from=args.init_from,
                        train_layers=(tuple(int(x) for x in args.train_layers.split(":"))
                                      if args.train_layers else None),
                        val_frac=args.val_frac, eval_every=args.eval_every,
                        resume=args.resume)
        )
    elif args.cmd == "serve":
        from .server import main as serve_main

        sys.argv = ["stateswap", "--model", args.model, "--persona-dir", args.persona_dir,
                    "--vocab", args.vocab, "--host", args.host, "--port", str(args.port)]
        serve_main()
    elif args.cmd == "chat":
        from .chat import run_chat

        run_chat(args.model, args.vocab, args.persona)


if __name__ == "__main__":
    main()
