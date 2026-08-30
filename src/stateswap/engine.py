"""Serving engine: persona states, session state-cache management, and
token-by-token generation.

核心卖点：
- 一个底座模型 + N 个人格（每人格一个 (L, H, 64, 64) 的 S0 张量，几 MB）。
- 每个会话持有自己的递归状态 Cache（RWKV 的 O(1) 状态 vs Transformer 的
  O(T) KV cache），多轮对话无需重放历史。
- 人格热切换 = 用新 S0 重建状态缓存，微秒级，不触碰权重。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch

from fla.models.utils import Cache

from .s0 import S0, load_base_model, make_cache
from .tokenizer import WorldTokenizer, load_tokenizer


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
        self.device = device
        self.tok: WorldTokenizer = load_tokenizer(str(vocab))
        self.personas: dict[str, Persona] = {}
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.register_persona("none", None, {"description": "S0 = 0 冷启动基线"})
        hidden = self.model.config.hidden_size
        self._state_bytes_per_layer = None

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
            s0 = payload["s0"].float()
            meta = {**(payload.get("meta") or {}), **(meta or {})}
        return self.register_tensor(name, s0, meta)

    def register_tensor(self, name: str, s0: torch.Tensor, meta: dict | None = None) -> Persona:
        """直接注册一个 S0 张量（state 算术的产物走这里）。"""
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
        session_id = uuid.uuid4().hex[:12]
        cache = self._cache_for(self.personas[persona_name])
        session = Session(session_id=session_id, persona_name=persona_name, cache=cache)
        with self._lock:
            self.sessions[session_id] = session
        return session

    def swap_persona(self, session_id: str, persona_name: str, keep_context: bool = False) -> dict:
        """O(1) 热切换：仅重建递归状态，不触碰任何权重。keep_context=True 时
        保留 token-shift 缓存（文本上下文），只换 S0。"""
        t0 = time.perf_counter()
        session = self.sessions[session_id]
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
        return {"swapped_to": persona_name, "keep_context": keep_context, "latency_ms": (time.perf_counter() - t0) * 1000}

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
        usage stats and no delta."""
        session = self.sessions[session_id]
        prompt = "User: " + user_text + "\n\nAssistant:"
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
        t_decode0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                nxt = self._sample(out.logits[0, -1], temperature, top_p, recent, rep_penalty)
                generated.append(nxt)
                recent.append(nxt)
                text = self.tok.decode(generated)
                if text.endswith("\n\n"):
                    break
                if text.endswith("User:"):
                    generated.pop()
                    break
                if text.endswith("User"):
                    # 停止符的前缀 token：不下发也不喂回，直接终止
                    generated.pop()
                    break
                yield {"delta": self.tok.decode([nxt])}
                out = self.model(
                    input_ids=torch.tensor([[nxt]], device=self.device),
                    past_key_values=session.cache,
                    use_cache=True,
                )
        decode_ms = (time.perf_counter() - t_decode0) * 1000
        reply = self.tok.decode(generated).split("\n\n")[0]
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
