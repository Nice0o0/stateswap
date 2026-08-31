"""Serving engine: persona states, session state-cache management, and
token-by-token generation.

核心卖点：
- 一个底座模型 + N 个人格（每人格一个 (L, H, 64, 64) 的 S0 张量，几 MB）。
- 每个会话持有自己的递归状态 Cache（RWKV 的 O(1) 状态 vs Transformer 的
  O(T) KV cache），多轮对话无需重放历史。
- 人格热切换 = 用新 S0 重建状态缓存，微秒级，不触碰权重。

并发约定：会话状态（递归 cache）在同一时刻只允许一个生成任务写入——
每个 Session 持有一把非阻塞锁，第二个并发请求会立刻收到 busy 错误，
而不是悄悄污染状态。
"""

from __future__ import annotations

import codecs
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fla.models.utils import Cache

from .s0 import S0, load_base_model, make_cache
from .tokenizer import WorldTokenizer, load_tokenizer

MAX_SESSIONS = 64


class SessionBusy(RuntimeError):
    """同一会话已有生成任务在写状态。"""


@dataclass
class Persona:
    name: str
    s0: torch.Tensor  # (L, H, K, V) fp32, on device
    meta: dict = field(default_factory=dict)

    @property
    def size_mb(self) -> float:
        return self.s0.numel() * self.s0.element_size() / 1e6


@dataclass
class Session:
    session_id: str
    persona_name: str
    cache: Cache
    history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    turns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_used: float = field(default_factory=time.time)

    def memory_mb(self, model) -> float:
        """Per-session state footprint in MB (recurrent + conv/ffn caches)."""
        total = 0
        for i in range(model.config.num_hidden_layers):
            st = self.cache[i]
            for key in ("recurrent_state", "conv_state", "ffn_state"):
                t = st.get(key)
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total / 1e6


class Engine:
    def __init__(
        self,
        model_dir: str,
        vocab: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = load_base_model(model_dir, device=device, dtype=dtype)
        self.model.eval()
        self.model_dir = str(Path(model_dir).resolve())
        self.device = device
        self.tok: WorldTokenizer = load_tokenizer(str(vocab))
        self.personas: dict[str, Persona] = {}
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.register_persona("none", None, {"description": "S0 = 0 冷启动基线"})

    # ---------------- persona management ----------------

    def register_persona(self, name: str, s0_path: str | Path | None, meta: dict | None = None) -> Persona:
        if s0_path is None:
            s0 = torch.zeros(
                self.model.config.num_hidden_layers,
                self.model.config.hidden_size // self.model.config.head_dim,
                self.model.config.head_dim,
                self.model.config.head_dim,
            )
        else:
            payload = torch.load(s0_path, map_location="cpu", weights_only=False)
            if "q" in payload:  # int8 量化人格（quant.save_quantized 产出）
                from .quant import dequantize_int8

                s0 = dequantize_int8(payload)
                meta = {**(payload.get("meta") or {}), "quantized": "int8", **(meta or {})}
            else:
                s0 = payload["s0"].float()
                meta = {**(payload.get("meta") or {}), **(meta or {})}
        return self.register_tensor(name, s0, meta)

    def register_tensor(self, name: str, s0: torch.Tensor, meta: dict | None = None) -> Persona | None:
        """直接注册一个 S0 张量（state 算术的产物走这里）。
        形状与当前底座不符（别的规格模型训的 S₀）时拒绝注册并返回 None。"""
        expected = (
            self.model.config.num_hidden_layers,
            self.model.config.hidden_size // self.model.config.head_dim,
            self.model.config.head_dim,
            self.model.config.head_dim,
        )
        if tuple(s0.shape) != expected:
            print(
                f"[stateswap] 跳过人格 {name!r}：S₀ 形状 {tuple(s0.shape)} 与当前底座 "
                f"{expected} 不匹配（其他规格模型训练的人格不能挂到这个底座）"
            )
            return None
        persona = Persona(name=name, s0=s0.float().to(self.device), meta=meta or {})
        with self._lock:
            self.personas[name] = persona
        return persona

    # ---------------- sessions ----------------

    def _cache_for(self, persona: Persona, batch_size: int = 1) -> Cache:
        holder = S0(self.model).to(self.device)
        holder.load_stacked(persona.s0)
        cache = make_cache(self.model, holder, batch_size=batch_size, detach_states=True)
        return cache

    def new_session(self, persona_name: str = "none") -> Session:
        if persona_name not in self.personas:
            raise KeyError(f"unknown persona: {persona_name}")
        self._prune_sessions()
        session_id = uuid.uuid4().hex[:12]
        cache = self._cache_for(self.personas[persona_name])
        session = Session(session_id=session_id, persona_name=persona_name, cache=cache)
        with self._lock:
            self.sessions[session_id] = session
        return session

    def _prune_sessions(self) -> None:
        """会话状态是常驻内存的（~6.4MB/个），超上限时回收最久未用的。"""
        with self._lock:
            if len(self.sessions) < MAX_SESSIONS:
                return
            oldest = min(self.sessions.values(), key=lambda s: s.last_used).session_id
            self.sessions.pop(oldest, None)

    def delete_persona(self, name: str) -> bool:
        with self._lock:
            return self.personas.pop(name, None) is not None

    def swap_persona(self, session_id: str, persona_name: str, keep_context: bool = False) -> dict:
        """O(1) 热切换：仅重建递归状态，不触碰任何权重。keep_context=True 时
        保留 token-shift 缓存（文本上下文），只换 S0。"""
        t0 = time.perf_counter()
        session = self.sessions[session_id]
        if not session.lock.acquire(blocking=False):
            raise SessionBusy(f"session {session_id} is busy")
        try:
            if persona_name not in self.personas:
                raise KeyError(f"unknown persona: {persona_name}")
            persona = self.personas[persona_name]
            if keep_context:
                # 只替换 recurrent_state，conv/ffn（词元级上下文）保留
                holder = S0(self.model).to(self.device)
                holder.load_stacked(persona.s0)
                new_state = make_cache(self.model, holder, detach_states=True)
                for i in range(self.model.config.num_hidden_layers):
                    session.cache.update(
                        recurrent_state=new_state[i]["recurrent_state"], layer_idx=i, offset=0
                    )
            else:
                session.cache = self._cache_for(persona)
                session.history = []
            session.persona_name = persona_name
            return {
                "swapped_to": persona_name,
                "keep_context": keep_context,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            }
        finally:
            session.lock.release()

    def drop_session(self, session_id: str) -> None:
        with self._lock:
            self.sessions.pop(session_id, None)

    # ---------------- generation ----------------

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, recent: list[int], rep_penalty: float) -> int:
        logits = logits.reshape(-1).float()
        if rep_penalty != 1.0 and recent:
            for t in set(recent[-128:]):
                logits[t] *= 1.0 if logits[t] < 0 else 1.0 / rep_penalty
        if temperature <= 1e-4:
            return int(logits.argmax())
        logits = logits / temperature
        if top_p < 1.0:
            sorted_ids = torch.argsort(logits, descending=True)
            sorted_probs = torch.softmax(logits, dim=-1)[sorted_ids]
            cutoff = torch.cumsum(sorted_probs, dim=-1)
            mask = cutoff - sorted_probs > top_p
            logits[sorted_ids[mask]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1))

    def chat(
        self,
        session_id: str,
        user_text: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        rep_penalty: float = 1.1,
    ) -> dict:
        chunks = list(
            self.chat_stream(
                session_id, user_text, max_new_tokens, temperature, top_p, rep_penalty
            )
        )
        return chunks[-1]

    def chat_stream(
        self,
        session_id: str,
        user_text: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        rep_penalty: float = 1.1,
    ):
        """Yield {"delta": text} as tokens are produced; the final yield carries
        usage stats and no delta.

        流式输出用增量 UTF-8 解码：World 词表是字节级的，颜文字/emoji/
        生僻字会拆成多个 token，逐 token decode 会产出 U+FFFD 乱码，
        必须缓存不完整的字节序列等到补齐。"""
        session = self.sessions[session_id]
        if not session.lock.acquire(blocking=False):
            raise SessionBusy(f"session {session_id} is busy")
        try:
            yield from self._chat_stream_locked(
                session, user_text, max_new_tokens, temperature, top_p, rep_penalty
            )
        finally:
            session.lock.release()

    def _chat_stream_locked(
        self,
        session: Session,
        user_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        rep_penalty: float,
    ):
        session.last_used = time.time()
        # 与训练侧 build_prompt 严格一致（含尾随空格）：S₀ 对 token 边界极其敏感
        prompt = "User: " + user_text + "\n\nAssistant: "
        prompt_ids = self.tok.encode(prompt)
        t_prefill0 = time.perf_counter()
        out = self.model(
            input_ids=torch.tensor([prompt_ids], device=self.device),
            past_key_values=session.cache,
            use_cache=True,
        )
        prefill_ms = (time.perf_counter() - t_prefill0) * 1000

        generated: list[int] = []
        recent: list[int] = list(prompt_ids)
        # 增量解码器：产出可解码字符的增量，未完整的多字节序列留在缓冲
        inc = codecs.getincrementaldecoder("utf-8")(errors="replace")
        full_text = ""
        t_decode0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                nxt = self._sample(out.logits[0, -1], temperature, top_p, recent, rep_penalty)
                generated.append(nxt)
                recent.append(nxt)
                delta = inc.decode(self.tok.id_to_bytes[nxt])
                full_text += delta
                stop = False
                if full_text.endswith("\n\n"):
                    stop = True
                elif full_text.endswith("User:"):
                    generated.pop()
                    stop = True
                elif full_text.endswith("User"):
                    # 停止符的前缀 token：不下发也不喂回
                    generated.pop()
                    full_text = full_text[: len(full_text) - len(delta)]
                    stop = True
                if stop:
                    break
                if delta:
                    yield {"delta": delta}
                out = self.model(
                    input_ids=torch.tensor([[nxt]], device=self.device),
                    past_key_values=session.cache,
                    use_cache=True,
                )
        decode_ms = (time.perf_counter() - t_decode0) * 1000
        reply = full_text.split("\n\n")[0]
        session.turns += 1
        session.history.append({"role": "user", "content": user_text})
        session.history.append({"role": "assistant", "content": reply})
        yield {
            "reply": reply,
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated),
            "prefill_ms": round(prefill_ms, 2),
            "decode_ms_per_token": round(decode_ms / max(1, len(generated)), 2),
            "session_memory_mb": round(session.memory_mb(self.model), 3),
            "turns": session.turns,
        }

    def batch_chat(
        self,
        session_ids: list[str],
        user_texts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[dict]:
        """多会话批量解码：把 B 个会话的递归状态沿 batch 维堆叠，
        一次前向推进 B 个会话。RWKV 无 attention，batch 成本近似线性，
        吞吐随会话数近线性扩展（对比逐会话循环）。

        返回每个会话的 {"reply", "completion_tokens"} 列表（顺序同输入）。"""
        if len(session_ids) != len(user_texts):
            raise ValueError("session_ids and user_texts length mismatch")
        sessions = [self.sessions[sid] for sid in session_ids]
        acquired: list[Session] = []
        try:
            for s in sessions:
                if not s.lock.acquire(blocking=False):
                    raise SessionBusy(f"session {s.session_id} is busy")
                acquired.append(s)

            B = len(sessions)
            L = self.model.config.num_hidden_layers

            # 1) 逐会话预填充（长度不同，各自 B=1 prefill）
            prefill_logits = []
            for s, text in zip(sessions, user_texts):
                prompt = "User: " + text + "\n\nAssistant: "
                ids = self.tok.encode(prompt)
                out = self.model(
                    input_ids=torch.tensor([ids], device=self.device),
                    past_key_values=s.cache, use_cache=True,
                )
                prefill_logits.append(out.logits[0, -1])
                s.last_used = time.time()

            # 2) 状态沿 batch 维堆叠成共享 Cache
            stacked = Cache()
            for i in range(L):
                stacked.update(
                    recurrent_state=torch.cat(
                        [s.cache[i]["recurrent_state"] for s in sessions], dim=0),
                    conv_state=torch.cat(
                        [s.cache[i]["conv_state"] for s in sessions], dim=0),
                    ffn_state=torch.cat(
                        [s.cache[i]["ffn_state"] for s in sessions], dim=0),
                    layer_idx=i, offset=0,
                )

            # 3) 批量解码循环：已完成的会话继续喂数据保持状态推进，输出忽略
            generated = [[] for _ in range(B)]
            finished = [False] * B
            next_ids = torch.tensor(
                [[int(logits.argmax())] for logits in prefill_logits], device=self.device)
            for b in range(B):
                generated[b].append(int(next_ids[b, 0]))

            with torch.no_grad():
                for _ in range(max_new_tokens - 1):
                    if all(finished):
                        break
                    out = self.model(input_ids=next_ids, past_key_values=stacked, use_cache=True)
                    next_ids = torch.empty((B, 1), dtype=torch.long, device=self.device)
                    for b in range(B):
                        if finished[b]:
                            next_ids[b, 0] = generated[b][-1]
                            continue
                        nxt = self._sample(out.logits[b, -1], temperature, top_p,
                                           generated[b], 1.0)
                        generated[b].append(nxt)
                        next_ids[b, 0] = nxt
                        if self.tok.decode(generated[b]).endswith("\n\n"):
                            finished[b] = True

            # 4) 把批量状态写回各会话
            for b, s in enumerate(sessions):
                for i in range(L):
                    st = stacked[i]
                    s.cache.update(
                        recurrent_state=st["recurrent_state"][b:b + 1].contiguous(),
                        conv_state=st["conv_state"][b:b + 1].contiguous(),
                        ffn_state=st["ffn_state"][b:b + 1].contiguous(),
                        layer_idx=i, offset=0,
                    )

            results = []
            for b, s in enumerate(sessions):
                reply = self.tok.decode(generated[b]).split("\n\n")[0].strip()
                s.turns += 1
                s.history.append({"role": "user", "content": user_texts[b]})
                s.history.append({"role": "assistant", "content": reply})
                results.append({"reply": reply, "completion_tokens": len(generated[b])})
            return results
        finally:
            for s in acquired:
                s.lock.release()

    # ---------------- stateless OpenAI-style completion ----------------

    def complete_from_messages(
        self, persona_name: str, messages: list[dict], max_new_tokens: int = 512,
        temperature: float = 1.0, top_p: float = 0.9,
    ) -> dict:
        """无会话模式：每请求把完整 messages 重新 prefill（Transformer 式）。
        与 session 模式对比即可展示 RWKV 状态缓存的收益。"""
        session = self.new_session(persona_name)
        try:
            parts = []
            for m in messages[:-1]:
                if m["role"] in ("user", "assistant"):
                    parts.append(("User: " if m["role"] == "user" else "") + m["content"] + "\n\n")
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            text = "".join(parts) + last_user
            result = self.chat(
                session.session_id,
                text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        finally:
            self.drop_session(session.session_id)
        return result
