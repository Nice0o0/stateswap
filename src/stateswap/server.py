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
import io
import json
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .engine import MAX_CONTEXT_REPLAY, MAX_NEW_TOKENS_LIMIT, Engine, SessionBusy
from .safety import content_disposition, safe_join, validate_name

MAX_CONTEXT_REPLAY_LIMIT = MAX_CONTEXT_REPLAY


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
    # 小模型 + 风格化人格在高温下会语无伦次，默认取平衡档。
    # 全部带约束：这些值直接进采样器与 GPU 循环，rep_penalty=0 会让
    # 1/rep_penalty 抛 ZeroDivisionError，无上限的 max_tokens 能占住显卡数分钟。
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.8, gt=0.0, le=1.0)
    max_tokens: int = Field(512, ge=1, le=MAX_NEW_TOKENS_LIMIT)
    rep_penalty: float = Field(1.25, gt=0.0, le=3.0)
    no_repeat_ngram: int = Field(8, ge=0, le=16)


class SessionRequest(BaseModel):
    persona: str = "none"
    # 有界文本重放（轮数）：近期事实以可见文本存在，弥补 1.5B 状态事实保持的
    # 不足（实测即刻回忆即失败）；K 有界仍是 O(1)，0 = 纯状态模式
    context_replay: int = Field(4, ge=0, le=MAX_CONTEXT_REPLAY_LIMIT)
    # 吸引子种子：True=人格当场生成一条种子交换写入状态（锁定吸引子、
    # 不对用户显示）；字符串=显式助手种子文本；False=不播种
    seed: bool | str = False


class SwapRequest(BaseModel):
    persona: str
    keep_context: bool = False


class RegisterRequest(BaseModel):
    name: str
    path: str  # 服务器进程可见的人格文件路径（s0.pt / int8 / rank4 均可）


class MixRequest(BaseModel):
    a: str
    b: str
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    name: str


class TrainRequest(BaseModel):
    data: str
    persona_name: str
    steps: int = Field(800, ge=10, le=20000)
    lr: float = Field(1e-4, gt=0.0, le=1e-2)
    ctx: int = Field(512, ge=64, le=4096)
    turns: int = Field(1, ge=1, le=8)  # >1 把若干对拼成多轮链（长对话人格的推荐训练法）


@dataclass(frozen=True)
class ProjectPaths:
    """项目里被服务层读写的目录（统一解析，避免 __file__ 在 wheel 安装下指错）。

    显式传 root 时所有子目录都相对它；不传时优先用当前工作目录（仓库布局），
    退回到包所在位置的上两级。
    """

    root: Path | None = None

    def __post_init__(self) -> None:
        root = self.root
        if root is None:
            cwd = Path.cwd()
            root = cwd if (cwd / "src" / "stateswap").is_dir() or (cwd / "web").is_dir() \
                else Path(__file__).resolve().parents[2]
        object.__setattr__(self, "root", Path(root).resolve())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def personas(self) -> Path:
        return self.root / "personas"

    @property
    def vocab(self) -> Path:
        return self.root / "vendor" / "rwkv_vocab_v20230424.txt"

    @property
    def web(self) -> Path:
        return self.root / "web"


def create_app(
    model_dir: str,
    vocab: str,
    persona_dirs: list[str] | None = None,
    persona_dir: str | None = None,
    paths: ProjectPaths | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # startup：加载底座并注册人格（on_event 已被 FastAPI 弃用）
        root = Path(persona_dir) if persona_dir else None
        engine = Engine(model_dir, vocab, personas_root=root)
        # 兼容两种来源：--persona-dir 下的目录（每个含 s0.pt），或显式列表
        candidates: list[Path] = []
        if persona_dirs:
            candidates = [Path(p) for p in persona_dirs]
        elif persona_dir:
            pdir = Path(persona_dir)
            if not pdir.is_dir():
                print(f"[stateswap] 人格目录不存在，跳过自动加载: {pdir}")
            else:
                candidates = sorted(p for p in pdir.iterdir()
                                    if p.is_dir() and (p / "s0.pt").exists())
        # 单个坏文件不能拖垮整个服务：一个损坏/异规格的 s0.pt 以前会让 startup
        # 抛异常，服务直接起不来（所有人格都用不了）。逐个隔离，打印原因后跳过。
        def _try_register(name: str, path: Path) -> None:
            try:
                engine.register_persona(name, str(path))
            except Exception as e:  # noqa: BLE001
                print(f"[stateswap] 跳过无法加载的人格 {name}（{path}）: {type(e).__name__}: {e}")

        for p in candidates:
            _try_register(p.name, p / "s0.pt")
        # 顶层单文件人格：int8 量化（quant.save_quantized）与
        # 低秩分解（lowrank.save_lowrank，*.rank<N>.pt）产物
        if persona_dir and Path(persona_dir).is_dir():
            for p in sorted(Path(persona_dir).glob("*.int8.pt")):
                _try_register(p.name.replace(".int8.pt", "-int8"), p)
            for p in sorted(Path(persona_dir).glob("*.rank*.pt")):
                _try_register(p.name[:-3].replace(".rank", "-rank"), p)
        state["engine"] = engine
        yield
        # shutdown：状态在内存中，无需持久化

    app = FastAPI(title="stateswap", version="0.1.0", lifespan=lifespan)
    state = {"engine": None}
    paths = paths or ProjectPaths()

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        """OpenAI 风格错误体 {"error": {"message", "type", "code"}}——
        官方 SDK 与 WebUI 都能读；HTTP 状态码不变。"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error",
                    "code": exc.status_code,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        """请求体校验失败（422）也走 OpenAI 错误体，客户端处理逻辑统一。"""
        msgs = "; ".join(str(e.get("msg", "")) for e in exc.errors()[:3])
        return JSONResponse(
            status_code=422,
            content={"error": {"message": msgs or "invalid request body",
                               "type": "invalid_request_error", "code": 422}},
        )

    @app.middleware("http")
    async def no_cache_frontend(request, call_next):
        """前端文件不缓存（no-cache = 每次用 ETag/mtime revalidate：没变 304，
        变了立即拿新文件）——否则改了 web/ 之后浏览器还端着旧页面。"""
        response = await call_next(request)
        p = request.url.path
        if p == "/" or p.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache"
        return response

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
                    "size_mb": round(p.size_mb, 2),  # 文件大小（int8/rank4 为真实压缩后体积）
                    "memory_mb": round(p.memory_mb, 2),  # 内存态（fp32，恒 12.58）
                    "meta": p.meta,
                    "shape": list(p.s0.shape),
                }
                for p in engine.personas.values()
            ]
        }

    @app.get("/v1/models")
    def list_models():
        """OpenAI 兼容：很多客户端启动时会先列模型。model id = 人格名。"""
        engine = _engine()
        return {
            "object": "list",
            "data": [
                {"id": p.name, "object": "model", "created": 0, "owned_by": "stateswap"}
                for p in engine.personas.values()
            ],
        }

    @app.post("/v1/sessions")
    def create_session(req: SessionRequest = Body(...)):
        engine = _engine()
        try:
            session = engine.new_session(req.persona, context_replay=req.context_replay,
                                         seed=req.seed)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {
            "session_id": session.session_id,
            "persona": session.persona_name,
            "seeded": bool(req.seed),
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

    @app.post("/v1/personas/register")
    def register_persona(req: RegisterRequest = Body(...)):
        """运行时从磁盘注册人格，免重启（persona factory 管线的最后一步）。
        形状与底座不符时 400——其他规格模型训练的 S₀ 挂不上来。

        path 必须落在人格根目录内：否则远端可以注册任意本地 .pt，再用
        `/bundle` 把它打包外带（任意文件读取 / 路径探测）。
        """
        engine = _engine()
        try:
            name = validate_name(req.name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if name in engine.personas:
            raise HTTPException(409, f"persona {name} already exists")
        raw = Path(req.path)
        if not raw.is_file():
            raise HTTPException(404, f"persona file not found: {req.path}")
        root = engine.personas_root
        if root is not None and not raw.resolve().is_relative_to(root):
            raise HTTPException(
                400, f"persona file must live under {root}（只允许注册人格目录内的文件）"
            )
        try:
            persona = engine.register_persona(name, str(raw))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"cannot load persona file: {type(e).__name__}: {e}")
        if persona is None:  # register_tensor 已打印形状不匹配的原因
            raise HTTPException(400, "S0 shape does not match the loaded base model")
        return {"name": persona.name, "size_mb": round(persona.size_mb, 2), "meta": persona.meta}

    @app.get("/v1/personas/{name}/bundle")
    def export_bundle(name: str):
        """导出人格包（zip：s0.pt + meta.json + eval.json）——人格成为可分享资产。
        仅支持从磁盘文件注册的人格（mix 结果无文件来源）。"""
        engine = _engine()
        p = engine.personas.get(name)
        if p is None:
            raise HTTPException(404, f"unknown persona {name}")
        src = (p.meta or {}).get("source")
        if not src or not Path(src).is_file():
            raise HTTPException(400, "persona has no file source（mix 结果先保存再导出）")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("s0.pt", Path(src).read_bytes())
            meta = {k: v for k, v in (p.meta or {}).items() if k != "source"}
            meta["name"] = name
            z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
            ev = Path(src).parent / "eval.json"
            if ev.is_file():
                z.writestr("eval.json", ev.read_bytes())
            z.writestr(
                "README.txt",
                "stateswap persona bundle\ns0.pt + meta.json (+eval.json)\n"
                "导入：POST /v1/personas/import_bundle（raw body = 本 zip）\n",
            )
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/zip",
            # 中文名/引号/换行会让 latin-1 编码失败或破坏 header：统一走 RFC 5987
            headers={"Content-Disposition": content_disposition(f"{name}.persona.zip")},
        )

    # 单个 zip 条目的解压后上限：fp32 人格 12.6MB、rank4 1.6MB，20MB 足够
    MAX_ENTRY_BYTES = 20 * 1024 * 1024
    MAX_COMPRESSION_RATIO = 100

    @app.post("/v1/personas/import_bundle")
    async def import_bundle(request: Request):
        """导入人格包并注册（免重启）。zip 内只认固定文件名，杜绝 zip-slip。

        同时限制**解压后**体积与压缩比：只查压缩包大小挡不住 zip 炸弹
        （几百 KB 的 deflate 流可以声明解压出几百 MB）。
        """
        engine = _engine()
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body")
        if len(body) > 64 * 1024 * 1024:
            raise HTTPException(400, "bundle too large (max 64MB)")

        def _read(z: zipfile.ZipFile, entry: str) -> bytes | None:
            if entry not in z.namelist():
                return None
            info = z.getinfo(entry)
            if info.file_size > MAX_ENTRY_BYTES:
                raise ValueError(f"{entry} 解压后过大（{info.file_size} > {MAX_ENTRY_BYTES}）")
            if info.compress_size and info.file_size > MAX_COMPRESSION_RATIO * info.compress_size:
                raise ValueError(f"{entry} 压缩比异常（疑似 zip 炸弹）")
            return z.read(entry)

        try:
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                data = _read(z, "s0.pt")
                if data is None:
                    raise ValueError("missing s0.pt")
                meta_raw = _read(z, "meta.json")
                meta = json.loads(meta_raw) if meta_raw else {}
                eval_bytes = _read(z, "eval.json")
        except (zipfile.BadZipFile, ValueError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"invalid bundle: {e}")
        try:
            name = validate_name(str(meta.get("name") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if name in engine.personas:
            raise HTTPException(409, f"persona {name} already exists")
        root = engine.personas_root or (Path(engine.model_dir).parent.parent / "personas")
        try:
            out_dir = safe_join(root, name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "s0.pt").write_bytes(data)
        if eval_bytes:
            (out_dir / "eval.json").write_bytes(eval_bytes)
        try:
            persona = engine.register_persona(name, str(out_dir / "s0.pt"))
        except Exception as e:  # noqa: BLE001
            (out_dir / "s0.pt").unlink(missing_ok=True)
            raise HTTPException(400, f"cannot load bundle: {type(e).__name__}: {e}")
        if persona is None:
            (out_dir / "s0.pt").unlink(missing_ok=True)
            raise HTTPException(400, "S0 shape does not match the loaded base model")
        return {"name": persona.name, "size_mb": round(persona.size_mb, 2)}

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
    async def chat_completions(req: ChatRequest = Body(...), request: Request = None):
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

            async def _run():
                if req.stream:
                    async def sse():
                        # 协作式取消：同步生成器不能被事件循环线程 close()——
                        # 正在执行时会抛 "generator already executing"，锁要等到
                        # 被遗弃的 worker 线程跑完才释放。用 Event 让生成循环在
                        # 下一个 token 边界自己退出（毫秒级），锁随即释放。
                        cancel = threading.Event()
                        gen = engine.chat_stream(
                            session_id, last_user, req.max_tokens, req.temperature,
                            req.top_p, rep_penalty=req.rep_penalty,
                            no_repeat_ngram=req.no_repeat_ngram,
                            cancel=cancel,
                        )
                        try:
                            async for piece in iterate_in_threadpool(gen):
                                # 客户端断开（关页面/切走）→ 立即停止生成，不白烧 GPU
                                if request is not None and await request.is_disconnected():
                                    cancel.set()
                                    return
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
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": (
                                                    "length" if piece.get("truncated") else "stop"
                                                ),
                                            }],
                                            "stateswap": {
                                                "prefill_ms": piece["prefill_ms"],
                                                "decode_ms_per_token": piece["decode_ms_per_token"],
                                                "session_memory_mb": piece["session_memory_mb"],
                                                "degenerated": piece.get("degenerated", False),
                                                "truncated": piece.get("truncated", False),
                                            },
                                            "usage": {
                                                "prompt_tokens": piece["prompt_tokens"],
                                                "completion_tokens": piece["completion_tokens"],
                                                "total_tokens": piece["prompt_tokens"] + piece["completion_tokens"],
                                            },
                                        }
                                    ) + "\n\n"
                            yield "data: [DONE]\n\n"
                        except SessionBusy as e:
                            yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
                            yield "data: [DONE]\n\n"
                        finally:
                            # 兜底：任何路径（异常/断流）都置位，确保生成循环退出
                            cancel.set()
                            try:
                                gen.close()  # 已自行退出时是 no-op
                            except ValueError:  # 仍在执行：cancel 已置位，等它退出
                                pass
                    return StreamingResponse(sse(), media_type="text/event-stream")
                try:
                    result = await run_in_threadpool(
                        engine.chat,
                        session_id, last_user, req.max_tokens, req.temperature,
                        req.top_p, rep_penalty=req.rep_penalty,
                        no_repeat_ngram=req.no_repeat_ngram,
                    )
                except SessionBusy as e:
                    raise HTTPException(409, str(e))
                return _openai_response(chat_id, created, persona, result)

            return await _run()

        # 无会话：把完整 messages 重新 prefill（Transformer 式语义）
        if req.stream:
            raise HTTPException(400, "streaming requires session_id")
        try:
            result = await run_in_threadpool(
                engine.complete_from_messages,
                persona, messages, req.max_tokens, req.temperature, req.top_p,
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
                    # 触到 max_tokens 上限时必须是 "length"，SDK 靠这个判断截断
                    "finish_reason": "length" if result.get("truncated") else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "stateswap": {
                "prefill_ms": result["prefill_ms"],
                "decode_ms_per_token": result["decode_ms_per_token"],
                "session_memory_mb": result["session_memory_mb"],
                "degenerated": result.get("degenerated", False),
                "truncated": result.get("truncated", False),
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
        try:
            persona = engine.register_tensor(
                req.name, tensor, {"mix": {"a": req.a, "b": req.b, "alpha": req.alpha}}
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        if persona is None:
            raise HTTPException(400, "mixed S0 shape does not match the loaded base model")
        return {"name": persona.name, "size_mb": round(persona.size_mb, 2)}

    # ---------------- training from the WebUI ----------------

    TRAIN_STATE = {"running": False, "error": None, "persona": None}

    @app.get("/v1/train/datasets")
    def train_datasets():
        files = sorted(p.name for p in paths.data.glob("*.json")) if paths.data.is_dir() else []
        return {"datasets": files}

    @app.get("/v1/train/status")
    def train_status():
        return TRAIN_STATE

    @app.post("/v1/train/start")
    def train_start(req: TrainRequest = Body(...)):
        engine = _engine()
        if TRAIN_STATE["running"]:
            raise HTTPException(409, "a training run is already in progress")
        try:
            name = validate_name(req.persona_name)
        except ValueError as e:
            raise HTTPException(400, str(e))
        data_path = paths.data / Path(req.data).name
        if not data_path.is_file():
            raise HTTPException(404, f"dataset {req.data} not found")
        # 训练产物写进人格目录：同名人格会被整个覆盖（12.6MB 重训一次就没了）
        if name in engine.personas or (paths.personas / name).exists():
            raise HTTPException(
                409, f"persona {name} already exists（换一个名字训练，或先删除旧人格目录）"
            )

        def _worker():
            from .train import TrainConfig, train_s0

            TRAIN_STATE.update(
                running=True, error=None, step=0, total=req.steps, loss=None,
                lr=None, persona=name,
            )

            def progress(step, total, loss, lr, gnorm, s0_inf):
                TRAIN_STATE.update(step=step, total=total, loss=loss, lr=lr)

            try:
                out_dir = paths.personas / name
                cfg = TrainConfig(
                    model_dir=engine.model_dir,
                    vocab=str(paths.vocab),
                    data=str(data_path),
                    out=str(out_dir),
                    steps=req.steps,
                    lr=req.lr,
                    ctx=req.ctx if req.turns == 1 else max(req.ctx, 1024),
                    turns=req.turns,
                    log_every=max(10, req.steps // 50),
                    meta={"trained_via": "webui"},
                )
                train_s0(cfg, progress_fn=progress)
                engine.register_persona(name, str(out_dir / "s0.pt"))
                TRAIN_STATE.update(running=False, done=True, step=req.steps)
            except Exception as e:  # noqa: BLE001
                TRAIN_STATE.update(running=False, done=True, error=repr(e)[:300])

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True, "persona": name}

    # 静态 WebUI：放在最后注册，未匹配的路径（含 /）落到 web/ 目录
    web_dir = paths.web
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


def _default_paths() -> "ProjectPaths":
    """仓库布局的默认根：优先当前工作目录（wheel 安装时 __file__ 指向
    site-packages，用它拼 data/ 与 web/ 会找错地方）。"""
    cwd = Path.cwd()
    if (cwd / "src" / "stateswap").is_dir() or (cwd / "web").is_dir():
        return ProjectPaths(root=cwd)
    return ProjectPaths(root=Path(__file__).resolve().parents[2])


def main():
    ap = argparse.ArgumentParser(description="stateswap OpenAI-compatible server")
    ap.add_argument("--model", required=True, help="converted RWKV-7 model dir")
    ap.add_argument("--vocab", default=None, help="World vocab (default: <root>/vendor/...)")
    ap.add_argument("--persona-dir", default=None, help="dir containing <name>/s0.pt")
    ap.add_argument("--root", default=None,
                    help="项目根目录（data/ personas/ vendor/ web/ 相对它解析）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    paths = ProjectPaths(root=Path(args.root) if args.root else None)
    app = create_app(
        args.model, args.vocab or str(paths.vocab),
        persona_dir=args.persona_dir or str(paths.personas),
        paths=paths,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
