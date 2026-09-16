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
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import Engine, SessionBusy


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
    # 小模型 + 风格化人格在高温下会语无伦次，默认取平衡档
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 512
    rep_penalty: float = 1.25
    no_repeat_ngram: int = 8


class SessionRequest(BaseModel):
    persona: str = "none"


class SwapRequest(BaseModel):
    persona: str
    keep_context: bool = False


class MixRequest(BaseModel):
    a: str
    b: str
    alpha: float = 0.5
    name: str


class TrainRequest(BaseModel):
    data: str
    persona_name: str
    steps: int = 800
    lr: float = 1e-4
    ctx: int = 512


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
        # 顶层单文件人格：int8 量化（quant.save_quantized）与
        # 低秩分解（lowrank.save_lowrank，*.rank<N>.pt）产物
        if persona_dir:
            for p in sorted(Path(persona_dir).glob("*.int8.pt")):
                engine.register_persona(p.name.replace(".int8.pt", "-int8"), str(p))
            for p in sorted(Path(persona_dir).glob("*.rank*.pt")):
                engine.register_persona(p.name[:-3].replace(".rank", "-rank"), str(p))
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
                {
                    "name": p.name,
                    "size_mb": round(p.size_mb, 2),
                    "meta": p.meta,
                    "shape": list(p.s0.shape),
                }
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

    @app.get("/v1/sessions")
    def list_sessions():
        engine = _engine()
        sessions = sorted(engine.sessions.values(), key=lambda s: s.last_used, reverse=True)
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "persona": s.persona_name,
                    "turns": s.turns,
                    "memory_mb": round(s.memory_mb(engine.model), 3),
                    "last_used": s.last_used,
                }
                for s in sessions
            ]
        }

    @app.get("/v1/sessions/{session_id}")
    def session_detail(session_id: str):
        engine = _engine()
        s = engine.sessions.get(session_id)
        if s is None:
            raise HTTPException(404, f"unknown session {session_id}")
        return {
            "session_id": s.session_id,
            "persona": s.persona_name,
            "turns": s.turns,
            "memory_mb": round(s.memory_mb(engine.model), 3),
            "history": s.history,
        }

    @app.get("/v1/sessions/{session_id}/state-stats")
    def session_state_stats(session_id: str):
        """每层递归状态范数 + 与人格 S0 的余弦（WebUI 状态监视器数据源）。"""
        engine = _engine()
        if session_id not in engine.sessions:
            raise HTTPException(404, f"unknown session {session_id}")
        return engine.session_state_stats(session_id)

    @app.delete("/v1/personas/{name}")
    def delete_persona(name: str):
        """仅从注册表移除（内存态），磁盘上的 personas/<name>/s0.pt 不动。
        引用该人格的既有会话仍可继续对话（状态已复制进会话缓存）。"""
        if not _engine().delete_persona(name):
            raise HTTPException(404, f"unknown persona {name}")
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
        except SessionBusy as e:
            raise HTTPException(409, str(e))
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
                try:
                    engine.swap_persona(req.session_id, persona)
                except SessionBusy as e:
                    raise HTTPException(409, str(e))
            session_id = req.session_id
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )

            def _run():
                if req.stream:
                    def sse():
                        try:
                            for piece in engine.chat_stream(
                                session_id, last_user, req.max_tokens, req.temperature,
                                req.top_p, rep_penalty=req.rep_penalty,
                                no_repeat_ngram=req.no_repeat_ngram,
                            ):
                                if "delta" in piece:
                                    yield "data: " + json.dumps(
                                        {
                                            "id": chat_id, "object": "chat.completion.chunk",
                                            "created": created, "model": persona,
                                            "choices": [{"index": 0, "delta": {"content": piece["delta"]}}],
                                        }
                                    ) + "\n\n"
                                elif "reply" in piece:
                                    # 结束帧：附上延迟/内存统计，供前端展示
                                    yield "data: " + json.dumps(
                                        {
                                            "id": chat_id, "object": "chat.completion.chunk",
                                            "created": created, "model": persona,
                                            "choices": [{"index": 0, "delta": {}}],
                                            "stateswap": {
                                                "prefill_ms": piece["prefill_ms"],
                                                "decode_ms_per_token": piece["decode_ms_per_token"],
                                                "session_memory_mb": piece["session_memory_mb"],
                                                "degenerated": piece.get("degenerated", False),
                                            },
                                            "usage": {
                                                "prompt_tokens": piece["prompt_tokens"],
                                                "completion_tokens": piece["completion_tokens"],
                                            },
                                        }
                                    ) + "\n\n"
                            yield "data: [DONE]\n\n"
                        except SessionBusy as e:
                            yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
                            yield "data: [DONE]\n\n"
                    return StreamingResponse(sse(), media_type="text/event-stream")
                try:
                    result = engine.chat(
                        session_id, last_user, req.max_tokens, req.temperature,
                        req.top_p, rep_penalty=req.rep_penalty,
                        no_repeat_ngram=req.no_repeat_ngram,
                    )
                except SessionBusy as e:
                    raise HTTPException(409, str(e))
                return _openai_response(chat_id, created, persona, result)

            return _run()

        # 无会话：把完整 messages 重新 prefill（Transformer 式语义）
        if req.stream:
            raise HTTPException(400, "streaming requires session_id")
        try:
            result = engine.complete_from_messages(
                persona, messages, req.max_tokens, req.temperature, req.top_p
            )
        except SessionBusy as e:
            raise HTTPException(409, str(e))
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
                "degenerated": result.get("degenerated", False),
            },
        }

    # ---------------- state arithmetic: mix personas in-place ----------------

    @app.post("/v1/personas/mix")
    def mix_personas(req: MixRequest = Body(...)):
        """S = alpha·A + (1-alpha)·B，注册为新人格。注意：实测 S₀ 空间
        不可光滑插值（见 docs/state-arithmetic.md），混合体可能整体偏向
        某一个任务模式——这正是这个实验台的趣味所在。"""
        engine = _engine()
        for p in (req.a, req.b):
            if p not in engine.personas:
                raise HTTPException(404, f"unknown persona {p}")
        sa, sb = engine.personas[req.a].s0.shape, engine.personas[req.b].s0.shape
        if sa != sb:
            raise HTTPException(
                400,
                f"两个人格形状不同，无法混合（不同底座训练的 S₀）："
                f"{req.a}={list(sa)}，{req.b}={list(sb)}。"
                f"0.1B 底座的人格只能和 0.1B 的混，0.4B 和 0.4B 的混。",
            )
        if not req.name.strip():
            raise HTTPException(400, "persona name required")
        if req.name in engine.personas:
            raise HTTPException(409, f"persona {req.name} already exists")
        from .arithmetic import interpolate

        tensor = interpolate(engine.personas[req.a].s0, engine.personas[req.b].s0, req.alpha)
        engine.register_tensor(
            req.name, tensor, {"mix": {"a": req.a, "b": req.b, "alpha": req.alpha}}
        )
        return {"name": req.name, "size_mb": round(engine.personas[req.name].size_mb, 2)}

    # ---------------- training from the WebUI ----------------

    TRAIN_STATE = {"running": False, "error": None, "persona": None}

    @app.get("/v1/train/datasets")
    def train_datasets():
        root = Path(__file__).resolve().parents[2]
        data_dir = root / "data"
        files = sorted(p.name for p in data_dir.glob("*.json")) if data_dir.exists() else []
        return {"datasets": files}

    @app.get("/v1/train/status")
    def train_status():
        return TRAIN_STATE

    @app.post("/v1/train/start")
    def train_start(req: TrainRequest = Body(...)):
        engine = _engine()
        if TRAIN_STATE["running"]:
            raise HTTPException(409, "a training run is already in progress")
        if not req.persona_name.strip():
            raise HTTPException(400, "persona name required")
        root = Path(__file__).resolve().parents[2]
        data_path = root / "data" / Path(req.data).name
        if not data_path.exists():
            raise HTTPException(404, f"dataset {req.data} not found")

        def _worker():
            from .train import TrainConfig, train_s0

            TRAIN_STATE.update(
                running=True, error=None, step=0, total=req.steps, loss=None,
                lr=None, persona=req.persona_name,
            )

            def progress(step, total, loss, lr, gnorm, s0_inf):
                TRAIN_STATE.update(step=step, total=total, loss=loss, lr=lr)

            try:
                out_dir = root / "personas" / req.persona_name.strip()
                cfg = TrainConfig(
                    model_dir=engine.model_dir,
                    vocab=str(root / "vendor" / "rwkv_vocab_v20230424.txt"),
                    data=str(data_path),
                    out=str(out_dir),
                    steps=req.steps,
                    lr=req.lr,
                    ctx=req.ctx,
                    log_every=max(10, req.steps // 50),
                    meta={"trained_via": "webui"},
                )
                train_s0(cfg, progress_fn=progress)
                engine.register_persona(req.persona_name.strip(), str(out_dir / "s0.pt"))
                TRAIN_STATE.update(running=False, done=True)
            except Exception as e:  # noqa: BLE001
                TRAIN_STATE.update(running=False, done=True, error=repr(e)[:300])

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True, "persona": req.persona_name}

    # 静态 WebUI：放在最后注册，未匹配的路径（含 /）落到 web/ 目录
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

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
