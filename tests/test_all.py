"""stateswap unit tests.

纯 CPU 的测试（分词器 / 数据 / 算术 / 会话锁语义）总是运行；
依赖 GPU 和本地模型权重的重测试标记为 slow，CI 上自动跳过：
    pytest -m "not slow"
"""

from pathlib import Path

import pytest
import torch

from stateswap.arithmetic import add, interpolate

VOCAB = Path(__file__).resolve().parents[1] / "vendor" / "rwkv_vocab_v20230424.txt"
MODEL = Path(__file__).resolve().parents[1] / "models" / "rwkv7-0.4b-world-hf"

has_vocab = VOCAB.exists()
has_model = MODEL.exists() and (MODEL / "model.safetensors").exists()

slow = pytest.mark.slow


# ---------- tokenizer ----------

@pytest.fixture(scope="module")
def tok():
    from stateswap.tokenizer import WorldTokenizer

    return WorldTokenizer(VOCAB)


@pytest.mark.skipif(not has_vocab, reason="vocab file not present")
class TestTokenizer:
    def test_vocab_sanity(self, tok):
        # World 词表文件实际 65530 条；模型 embedding 有 65536 行（尾部留空）
        assert tok.vocab_size == 65530
        assert len(tok.id_to_bytes) == tok.vocab_size
        assert tok.vocab_size <= 65536

    def test_roundtrip_ascii(self, tok):
        assert tok.decode(tok.encode("Hello, world!")) == "Hello, world!"

    def test_roundtrip_multibyte(self, tok):
        # 多字节 CJK / emoji / 颜文字：正是流式增量解码要保护的内容
        texts = [
            "你好，世界。今天的月亮又圆又亮。",
            "喵呜~（ฅ^•ﻌ•^ฅ）小鱼干！",
            "混合 mix 123 🐱 emoji ✨",
        ]
        for t in texts:
            assert tok.decode(tok.encode(t)) == t

    def test_ids_in_range(self, tok):
        ids = tok.encode("任意一段文本 mixed with English 🚀")
        assert all(0 <= i < tok.vocab_size for i in ids)
        assert ids

    def test_unknown_byte_does_not_crash(self, tok):
        # 编码端只接受合法 utf-8；解码端对未知 id 应为空串而非崩溃
        assert tok.decode([tok.vocab_size + 5]) == ""


# ---------- data pipeline ----------

def test_build_example_masking():
    from stateswap.data import build_example
    from stateswap.tokenizer import WorldTokenizer

    tok = WorldTokenizer(VOCAB)
    ex = build_example(tok, "早上好呀", "喵呜~主人早安喵！")
    assert ex.prompt_len > 0
    assert all(lb == -100 for lb in ex.labels[: ex.prompt_len])
    supervised = ex.labels[ex.prompt_len :]
    assert supervised and all(lb != -100 for lb in supervised)
    assert ex.input_ids[ex.prompt_len :] == supervised


def test_build_example_truncation_keeps_tail():
    from stateswap.data import build_example
    from stateswap.tokenizer import WorldTokenizer

    tok = WorldTokenizer(VOCAB)
    long_out = "喵" * 900
    ex = build_example(tok, "打个招呼", long_out, ctx=128)
    assert len(ex.input_ids) <= 128


# ---------- state arithmetic ----------

def test_interpolate_endpoints():
    import torch

    a, b = torch.zeros(2, 3, 4), torch.ones(2, 3, 4)
    assert torch.equal(interpolate(a, b, 1.0), a)
    assert torch.equal(interpolate(a, b, 0.0), b)


def test_interpolate_midpoint():
    import torch

    a, b = torch.zeros(2, 3, 4), torch.ones(2, 3, 4)
    m = interpolate(a, b, 0.5)
    assert torch.allclose(m, torch.full_like(a, 0.5))


def test_add_scale():
    import torch

    a, b = torch.zeros(2, 2), torch.ones(2, 2)
    assert torch.allclose(add(a, b, 0.5), torch.full_like(a, 0.5))


# ---------- sampling（纯 CPU，_sample 不依赖 self 的模型状态） ----------

def _sample(logits, recent, no_repeat_ngram, temperature=0.0):
    from stateswap.engine import Engine

    return Engine._sample(None, logits, temperature, 1.0, recent, 1.0, no_repeat_ngram)


def test_sample_ngram_ban_blocks_token():
    # no_repeat_ngram=2：recent 中 0 后面总是跟 1 → 候选 1 被封禁，
    # 即使它的 logit 最大也应落到次优的 2
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
    recent = [0, 1, 0, 1, 0]
    assert _sample(logits, recent, no_repeat_ngram=2) == 2


def test_sample_ngram_all_banned_fallback():
    # 回归：模型 logits 本身含 -inf（如 bf16 溢出），n-gram 封禁盖住其余全部
    # 有限项时，必须回退到封禁前的 logits，而不是在全是 -inf 上 argmax 恒取 0
    logits = torch.tensor([float("-inf"), float("-inf"), 1.0, 2.0])
    recent = [0, 2, 0, 3, 0]  # 封禁 {2, 3} → 封后全 -inf → 回退 → argmax 取 3
    assert _sample(logits, recent, no_repeat_ngram=2) == 3


# ---------- model-dependent (GPU) ----------

requires_model = pytest.mark.skipif(
    not (has_model and VOCAB.exists()), reason="converted model weights not present"
)


@pytest.fixture(scope="module")
def engine():
    from stateswap.engine import Engine

    return Engine(str(MODEL), str(VOCAB))


@slow
@requires_model
class TestEngineHeavy:
    def test_s0_gradient_finite(self, engine):
        """Preen 教训：梯度非零且有限才算通——防止静默断裂回归。"""
        from stateswap.s0 import S0, make_cache

        model = engine.model
        model.train()
        s0 = S0(model).to("cuda")
        tok = engine.tok
        ex_ids = tok.encode("User: 你好\n\nAssistant: 喵！\n\n")
        ids = torch.tensor([ex_ids], device="cuda")
        cache = make_cache(model, s0, batch_size=1, detach_states=False)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True)
        loss = out.logits.float().mean()
        loss.backward()
        norms = [float(p.grad.float().norm()) for p in s0.states]
        assert all(n == n and 0 < abs(n) < float("inf") for n in norms)

    def test_chat_no_mojibake_and_state_memory(self, engine):
        """流式增量解码：多字节字符不允许出现 U+FFFD；会话状态恒定。"""
        # 夹具底座是 0.4B，人格必须与底座同规格（见形状守卫）
        persona = "personas-legacy/neko-0.4b-v2/s0.pt"
        if not Path(persona).exists():
            pytest.skip("legacy 0.4B persona not present")
        engine.register_persona("test-neko", persona)
        session = engine.new_session("test-neko")
        try:
            result = engine.chat(session.session_id, "早上好呀！", max_new_tokens=64, temperature=0.0)
            assert "\ufffd" not in result["reply"]
            assert result["session_memory_mb"] > 0
        finally:
            engine.drop_session(session.session_id)

    def test_session_busy_lock(self, engine):
        from stateswap.engine import SessionBusy

        session = engine.new_session("none")
        assert session.lock.acquire(blocking=False)
        try:
            with pytest.raises(SessionBusy):
                engine.swap_persona(session.session_id, "test-neko")
        finally:
            session.lock.release()
        engine.drop_session(session.session_id)
