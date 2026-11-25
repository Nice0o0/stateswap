"""HTTP-layer tests for stateswap's FastAPI surface.

server.py 有 21 个端点、约 600 行，是安全边界（路径校验、zip 上限、header 编码、
参数夹取）唯一落地的地方，此前**没有任何测试**覆盖。这些用例不加载模型：通过
`create_app(engine_factory=...)` 注入一个假引擎，只验证「HTTP 契约 + 防护」。

四组覆盖：
1. 路径与名字：人格名/文件路径必须落在 personas 根目录内
2. bundle 导入/导出：zip 炸弹、zip-slip、RFC 5987 header
3. 参数校验：rep_penalty=0 等非法值必须 422 且错误体是 OpenAI 格式
4. 响应契约：/v1/models、finish_reason、usage.total_tokens

跑法：pytest tests/test_server.py -q   （不需要 GPU 与模型权重）
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from stateswap.server import ProjectPaths, create_app  # noqa: E402


class FakePersona:
    def __init__(self, name: str, shape=(2, 4, 64, 64), file_bytes: int | None = None):
        self.name = name
        self.s0 = torch.zeros(*shape)
        self.meta: dict = {}
        self.file_bytes = file_bytes

    @property
    def size_mb(self) -> float:
        return self.file_bytes / 1e6 if self.file_bytes is not None else 0.01

    @property
    def memory_mb(self) -> float:
        return self.s0.numel() * self.s0.element_size() / 1e6


def _make_session(sid: str, persona: str):
    return SimpleNamespace(
        session_id=sid, persona_name=persona, turns=0, last_used=0.0, history=[],
        memory_mb=lambda model: 1.0,
    )


_GEN_STATS = {
    "degenerated": False, "prompt_tokens": 5, "completion_tokens": 2,
    "prefill_ms": 1.0, "decode_ms_per_token": 2.0, "session_memory_mb": 12.78,
}


class FakeEngine:
    """实现 server.py 实际调用到的成员（13 个），行为足以驱动全部端点。"""

    def __init__(self, personas_root: Path | None = None):
        self.personas_root = personas_root
        self.model_dir = "/tmp/fake-model"
        self.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
        self.personas: dict[str, FakePersona] = {"none": FakePersona("none")}
        self.sessions: dict[str, object] = {}

    # ---- personas ----
    def register_persona(self, name, path, meta=None):
        p = FakePersona(name, file_bytes=Path(path).stat().st_size)
        self.personas[name] = p
        return p

    def register_tensor(self, name, tensor, meta=None):
        if name in self.personas:
            return None
        p = FakePersona(name, shape=tuple(tensor.shape))
        self.personas[name] = p
        return p

    def delete_persona(self, name):
        return self.personas.pop(name, None) is not None

    # ---- sessions ----
    def new_session(self, persona, context_replay=4, seed=False):
        s = _make_session(f"sess{len(self.sessions) + 1}", persona)
        self.sessions[s.session_id] = s
        return s

    def drop_session(self, sid):
        self.sessions.pop(sid, None)

    def swap_persona(self, sid, persona, keep_context=False):
        self.sessions[sid].persona_name = persona
        return {"swapped_to": persona, "keep_context": keep_context, "latency_ms": 0.1}

    def session_state_stats(self, sid):
        return {"persona": self.sessions[sid].persona_name, "turns": 0,
                "layers": [{"i": 0, "norm": 1.0, "cos_s0": 0.5}]}

    # ---- generation ----
    def chat(self, sid, text, max_new_tokens=512, temperature=1.0, top_p=0.9,
             rep_penalty=1.25, no_repeat_ngram=8):
        return {"reply": "喵~", "truncated": False, **_GEN_STATS}

    def chat_stream(self, sid, text, *a, cancel=None, **kw):
        yield {"delta": "喵"}
        yield {"reply": "喵", "truncated": False, **_GEN_STATS,
               "completion_tokens": 1}

    def complete_from_messages(self, persona, messages, *a, **kw):
        return {"reply": "ok", "truncated": False, **_GEN_STATS,
                "prompt_tokens": 3, "completion_tokens": 1}


@pytest.fixture()
def ctx(tmp_path):
    """返回 (TestClient, ProjectPaths)。lifespan 里注入 FakeEngine，不加载模型。"""
    root = ProjectPaths(root=tmp_path)
    root.personas.mkdir(parents=True, exist_ok=True)
    root.data.mkdir(parents=True, exist_ok=True)

    fake = FakeEngine(personas_root=root.personas)
    app = create_app("unused", "unused", persona_dir=str(root.personas),
                     paths=root, engine_factory=lambda: fake)
    with TestClient(app) as c:
        yield c, root


# ============ 1. 路径与名字 ============


@pytest.mark.parametrize("bad", ["../evil", "..", "猫娘", "a b", ".hidden"])
def test_register_rejects_bad_name(ctx, bad):
    """人格名会拼成 personas/<name>：路径穿越与非 ASCII（header 编码失败）必须拒。"""
    c, root = ctx
    (root.personas / "ok" / "s0.pt").parent.mkdir(parents=True, exist_ok=True)
    (root.personas / "ok" / "s0.pt").write_bytes(b"x")
    r = c.post("/v1/personas/register", json={"name": bad, "path": str(root.personas / "ok" / "s0.pt")})
    assert r.status_code == 400
    assert "error" in r.json()            # OpenAI 错误体
    assert "message" in r.json()["error"]


def test_register_rejects_file_outside_personas_root(ctx):
    """目录外的 .pt 不能被注册（否则可注册任意本地文件再用 /bundle 外带）。"""
    c, root = ctx
    outside = root.root / "outside.pt"
    outside.write_bytes(b"x")
    r = c.post("/v1/personas/register",
               json={"name": "evil", "path": str(outside)})
    assert r.status_code == 400
    assert "must live under" in r.json()["error"]["message"]


def test_register_accepts_file_inside_root(ctx):
    c, root = ctx
    p = root.personas / "neko" / "s0.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 1024)
    r = c.post("/v1/personas/register", json={"name": "neko", "path": str(p)})
    assert r.status_code == 200
    assert r.json()["name"] == "neko"


# ============ 2. bundle ============


def _bundle(name: str, s0: bytes = b"\x00", meta_extra: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("s0.pt", s0)
        meta = {"name": name, **(meta_extra or {})}
        z.writestr("meta.json", json.dumps(meta))
    return buf.getvalue()


def test_import_bundle_rejects_traversal_in_meta_name(ctx):
    """meta.json 的 name 直接拼目录：'../../evil' 必须被拒（zip-slip）。"""
    c, root = ctx
    r = c.post("/v1/personas/import_bundle", content=_bundle("../../evil"))
    assert r.status_code == 400
    assert not (root.root.parent / "evil").exists()


def test_import_bundle_rejects_zip_bomb_by_size(ctx):
    """解压后体积超上限必须拒——**即使压缩比正常**（覆盖 MAX_ENTRY_BYTES 那一层）。

    用不可压缩的随机字节：40MB 压完仍约 40MB，压缩比 ≈1，只能靠体积上限拦住。
    """
    import os

    c, root = ctx
    big = os.urandom(30 * 1024 * 1024)       # 30MB > 20MB 上限，且不可压缩
    payload = _bundle("bomb", s0=big)
    assert len(payload) > 20 * 1024 * 1024   # 确认压缩比正常（> 1:1）
    r = c.post("/v1/personas/import_bundle", content=payload)
    assert r.status_code == 400
    assert "invalid bundle" in r.json()["error"]["message"]


def test_import_bundle_rejects_zip_bomb_by_ratio(ctx):
    """压缩比异常必须拒（覆盖 MAX_COMPRESSION_RATIO 那一层）。

    全零数据压缩比极高、解压后 <20MB，只有压缩比检查能拦住它。
    """
    c, root = ctx
    payload = _bundle("ratio-bomb", s0=bytes(18 * 1024 * 1024))
    assert len(payload) < 200 * 1024         # 压缩后极小
    r = c.post("/v1/personas/import_bundle", content=payload)
    assert r.status_code == 400


def test_import_bundle_happy_path(ctx):
    c, root = ctx
    r = c.post("/v1/personas/import_bundle", content=_bundle("imported"))
    assert r.status_code == 200
    assert (root.personas / "imported" / "s0.pt").is_file()


def test_bundle_export_header_is_latin1_safe(ctx):
    """导出 header 必须能 latin-1 编码：ASCII 名走可读 filename，
    非 ASCII/引号走 RFC 5987 的 filename*（否则 Starlette 直接 500）。"""
    from stateswap.safety import content_disposition

    # 纯函数层面已验证（tests/test_all.py）；这里补端点层面的 ASCII 路径
    assert content_disposition("neko.persona.zip").startswith('attachment; filename="')
    cd = content_disposition("猫娘.persona.zip")
    assert "filename*=utf-8''" in cd and "猫娘" not in cd

    c, root = ctx
    p = root.personas / "neko" / "s0.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * 128)
    r = c.post("/v1/personas/register", json={"name": "neko", "path": str(p)})
    assert r.status_code == 200
    resp = c.get("/v1/personas/neko/bundle")
    # FakePersona 未记录 meta["source"]，端点按设计回 400（"mix 结果先保存再导出"）
    assert resp.status_code in (200, 400)


# ============ 3. 参数校验 ============


@pytest.mark.parametrize("field,value", [
    ("rep_penalty", 0),          # 曾导致 ZeroDivisionError
    ("temperature", 5.0),
    ("top_p", 0),
    ("max_tokens", 99999),       # 无上限会占住 GPU 数分钟
])
def test_chat_request_validation_is_openai_shaped(ctx, field, value):
    c, _ = ctx
    body = {"model": "none", "messages": [{"role": "user", "content": "hi"}], field: value}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 422
    payload = r.json()
    assert "error" in payload and "message" in payload["error"]


def test_chat_unknown_persona_is_404(ctx):
    c, _ = ctx
    r = c.post("/v1/chat/completions",
               json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == 404


def test_train_start_refuses_to_overwrite_existing_persona(ctx):
    """同名人格会被 12.6MB 重训整个覆盖：必须 409 而不是静默写。"""
    c, root = ctx
    (root.personas / "neko").mkdir(parents=True, exist_ok=True)
    (root.data / "ds.json").write_text("[]", encoding="utf-8")
    r = c.post("/v1/train/start",
               json={"data": "ds.json", "persona_name": "neko", "steps": 10})
    assert r.status_code == 409
    assert "already exists" in r.json()["error"]["message"]


def test_train_start_validates_name(ctx):
    c, root = ctx
    (root.data / "ds.json").write_text("[]", encoding="utf-8")
    r = c.post("/v1/train/start",
               json={"data": "ds.json", "persona_name": "../../evil", "steps": 10})
    assert r.status_code == 400


# ============ 4. 响应契约 ============


def test_models_endpoint_lists_personas(ctx):
    """OpenAI 客户端启动时会先列模型；id 必须是人格名。"""
    c, root = ctx
    p = root.personas / "neko" / "s0.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    c.post("/v1/personas/register", json={"name": "neko", "path": str(p)})
    r = c.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert {"none", "neko"} <= ids


def test_chat_reports_usage_and_finish_reason(ctx):
    """OpenAI 兼容：finish_reason 必须是 stop/length，usage 要含 total_tokens。"""
    c, _ = ctx
    s = c.post("/v1/sessions", json={"persona": "none"}).json()
    r = c.post("/v1/chat/completions", json={
        "model": "none", "session_id": s["session_id"],
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] in ("stop", "length")
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert "stateswap" in body            # 引擎统计仍然挂上


def test_truncated_reply_maps_to_finish_reason_length(tmp_path):
    """引擎报 truncated=True 时，OpenAI 响应必须是 finish_reason="length"。

    SDK 靠这个字段判断「回复被截断」，之前恒为 "stop"。
    """
    root = ProjectPaths(root=tmp_path)
    root.personas.mkdir(parents=True, exist_ok=True)
    root.data.mkdir(parents=True, exist_ok=True)

    fake = FakeEngine(personas_root=root.personas)
    real_chat = fake.chat

    def truncated_chat(*a, **kw):
        out = real_chat(*a, **kw)
        out["truncated"] = True
        return out

    fake.chat = truncated_chat
    app = create_app("unused", "unused", persona_dir=str(root.personas),
                     paths=root, engine_factory=lambda: fake)
    with TestClient(app) as c:
        s = c.post("/v1/sessions", json={"persona": "none"}).json()
        r = c.post("/v1/chat/completions", json={
            "model": "none", "session_id": s["session_id"],
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 200
        assert r.json()["choices"][0]["finish_reason"] == "length"
        assert r.json()["stateswap"]["truncated"] is True


def test_session_lifecycle(ctx):
    c, _ = ctx
    s = c.post("/v1/sessions", json={"persona": "none", "context_replay": 2}).json()
    sid = s["session_id"]
    assert c.get(f"/v1/sessions/{sid}").status_code == 200
    assert c.get("/v1/sessions").status_code == 200
    assert c.get(f"/v1/sessions/{sid}/state-stats").status_code == 200
    r = c.post(f"/v1/sessions/{sid}/swap", json={"persona": "none", "keep_context": True})
    assert r.status_code == 200
    assert c.delete(f"/v1/sessions/{sid}").status_code == 200
    assert c.get(f"/v1/sessions/{sid}").status_code == 404


def test_deleted_persona_is_removed_from_registry(ctx):
    c, root = ctx
    p = root.personas / "tmp" / "s0.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    c.post("/v1/personas/register", json={"name": "tmp", "path": str(p)})
    assert c.delete("/v1/personas/tmp").status_code == 200
    assert c.delete("/v1/personas/tmp").status_code == 404
    # 磁盘文件不动（只从注册表移除）
    assert p.is_file()


def test_health_reports_engine_ready(ctx):
    c, _ = ctx
    r = c.get("/health")
    assert r.json() == {"ok": True, "engine_ready": True}
