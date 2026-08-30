"""OpenAI-compatible FastAPI server with persona hot-swapping.

Endpoints:
- POST /v1/chat/completions   OpenAI 格式；model 字段 = persona 名
- GET  /v1/personas           人格列表（含 S0 大小）
- POST /v1/sessions           新建会话（O(1) 多轮，不重放历史）
- POST /v1/sessions/{id}/swap 人格热切换
- DELETE /v1/sessions/{id}    销毁会话
- GET  /health
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .engine import Engine


# 请求模型必须定义在模块级：`from __future__ import annotations` 会把注解变成
# 字符串，函数内定义的类无法被 FastAPI 的 get_type_hints 解析（ForwardRef 报错）。
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "none"
    messages: list[ChatMessage]
    session_id: str | None = None
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 512


class SessionRequest(BaseModel):
    persona: str = "none"


class SwapRequest(BaseModel):
    persona: str
    keep_context: bool = False


def create_app(
    model_dir: str,
    vocab: str,
    persona_dirs: list[str] | None = None,
    persona_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title="stateswap", version="0.1.0")
    state = {"engine": None}

    @app.on_event("startup")
    def _startup():
        engine = Engine(model_dir, vocab)
        # 兼容两种来源：--persona-dir 下的目录（每个含 s0.pt），或显式列表
        candidates: list[Path] = []
        if persona_dirs:
            candidates = [Path(p) for p in persona_dirs]
        elif persona_dir:
            candidates = sorted(p for p in Path(persona_dir).iterdir() if (p / "s0.pt").exists())
        for p in candidates:
            engine.register_persona(p.name, str(p / "s0.pt"))
        state["engine"] = engine

    @app.get("/health")
    def health():
        return {"ok": True, "engine_ready": state["engine"] is not None}

    def _engine() -> Engine:
        if state["engine"] is None:
            raise HTTPException(503, "engine is still loading")
        return state["engine"]

    @app.get("/v1/personas")
    def list_personas():
        engine = _engine()
        return {
            "personas": [
                {"name": p.name, "size_mb": round(p.size_mb, 2), "meta": p.meta}
                for p in engine.personas.values()
            ]
        }

    @app.post("/v1/sessions")
    def create_session(req: SessionRequest = Body(...)):
        engine = _engine()
        try:
            session = engine.new_session(req.persona)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {
            "session_id": session.session_id,
            "persona": session.persona_name,
            "memory_mb": round(session.memory_mb(engine.model), 3),
        }

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str):
        _engine().drop_session(session_id)
        return {"ok": True}

    @app.post("/v1/sessions/{session_id}/swap")
    def swap(session_id: str, req: SwapRequest = Body(...)):
        engine = _engine()
        if session_id not in engine.sessions:
            raise HTTPException(404, f"unknown session {session_id}")
        try:
            result = engine.swap_persona(
                session_id, req.persona, keep_context=req.keep_context
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        session = engine.sessions[session_id]
        result["memory_mb"] = round(session.memory_mb(engine.model), 3)
        return result

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest = Body(...)):
        engine = _engine()
        persona = req.model
        if persona not in engine.personas:
            raise HTTPException(404, f"unknown persona {persona}")
        created = int(time.time())
        chat_id = "chatcmpl-" + uuid.uuid4().hex[:12]
        messages = [m.model_dump() for m in req.messages]

        if req.session_id:
            if req.session_id not in engine.sessions:
                raise HTTPException(404, f"unknown session {req.session_id}")
            if engine.sessions[req.session_id].persona_name != persona:
                engine.swap_persona(req.session_id, persona)
            session_id = req.session_id
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )

            def _run():
                if req.stream:
                    def sse():
                        for piece in engine.chat_stream(
                            session_id, last_user, req.max_tokens, req.temperature, req.top_p
                        ):
                            if "delta" in piece:
                                yield "data: " + json.dumps(
                                    {
                                        "id": chat_id, "object": "chat.completion.chunk",
                                        "created": created, "model": persona,
                                        "choices": [{"index": 0, "delta": {"content": piece["delta"]}}],
                                    }
                                ) + "\n\n"
                        yield "data: [DONE]\n\n"
                    return StreamingResponse(sse(), media_type="text/event-stream")
                result = engine.chat(
                    session_id, last_user, req.max_tokens, req.temperature, req.top_p
                )
                return _openai_response(chat_id, created, persona, result)

            return _run()

        # 无会话：把完整 messages 重新 prefill（Transformer 式语义）
        if req.stream:
            raise HTTPException(400, "streaming requires session_id")
        result = engine.complete_from_messages(
            persona, messages, req.max_tokens, req.temperature, req.top_p
        )
        return _openai_response(chat_id, created, persona, result)

    def _openai_response(chat_id, created, model, result):
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["reply"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
            },
            "stateswap": {
                "prefill_ms": result["prefill_ms"],
                "decode_ms_per_token": result["decode_ms_per_token"],
                "session_memory_mb": result["session_memory_mb"],
            },
        }

    return app


def main():
    ap = argparse.ArgumentParser(description="stateswap OpenAI-compatible server")
    ap.add_argument("--model", required=True, help="converted RWKV-7 model dir")
    ap.add_argument("--vocab", default="vendor/rwkv_vocab_v20230424.txt")
    ap.add_argument("--persona-dir", default="personas", help="dir containing <name>/s0.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = create_app(args.model, args.vocab, persona_dir=args.persona_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
