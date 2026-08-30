"""Interactive terminal chat with a persona state."""

from __future__ import annotations

import sys
from pathlib import Path

from .engine import Engine


def run_chat(model_dir: str, vocab: str, persona_name: str) -> None:
    engine = Engine(model_dir, vocab)
    if persona_name != "none":
        direct = Path(persona_name)
        packaged = Path("personas") / persona_name / "s0.pt"
        if packaged.exists():
            engine.register_persona(persona_name, str(packaged))
        elif direct.is_file() and direct.suffix == ".pt":
            engine.register_persona(direct.parent.name, str(direct))
            persona_name = direct.parent.name
    session = engine.new_session(persona_name)
    print(f"stateswap chat | persona={persona_name} | session={session.session_id}")
    print("commands: /persona <name>  /new  /exit")
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user.startswith("/persona"):
            _, _, name = user.partition(" ")
            print(engine.swap_persona(session.session_id, name.strip() or "none"))
            continue
        if user == "/new":
            session = engine.new_session(session.persona_name)
            print(f"new session {session.session_id}")
            continue
        sys.stdout.write("AI> ")
        for piece in engine.chat_stream(session.session_id, user):
            if "delta" in piece:
                sys.stdout.write(piece["delta"])
                sys.stdout.flush()
            else:
                stats = piece
        print(f"\n   [{stats['completion_tokens']} tok | prefill {stats['prefill_ms']}ms | "
              f"{stats['decode_ms_per_token']}ms/tok | state {stats['session_memory_mb']}MB]")
