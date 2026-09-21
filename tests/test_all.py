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


def test_build_multiturn_masks_all_assistant_spans():
    from stateswap.data import build_multiturn_example, build_prompt
    from stateswap.tokenizer import WorldTokenizer

    tok = WorldTokenizer(VOCAB)
    turns = [("早上好", "喵呜~主人早安！"), ("讲个笑话", "小鱼干走进了酒吧…喵！"), ("晚安", "好梦喵~")]
    ex = build_multiturn_example(tok, turns, ctx=1024)
    labels = ex.labels
    assert len(labels) == len(ex.input_ids)
    # 每个 Assistant 段都被监督（含结尾 \n\n），User 段全部掩码
    pos = 0
    for instruction, output in turns:
        p_len = len(tok.encode(build_prompt(instruction)))
        c_ids = tok.encode(output + "\n\n")
        assert all(lb == -100 for lb in labels[pos : pos + p_len])
        assert labels[pos + p_len : pos + p_len + len(c_ids)] == c_ids
        pos += p_len + len(c_ids)
    assert pos == len(ex.input_ids)


def test_build_multiturn_truncation_keeps_alignment():
    from stateswap.data import build_multiturn_example
    from stateswap.tokenizer import WorldTokenizer

    tok = WorldTokenizer(VOCAB)
    turns = [("你好", "喵" * 200), ("继续", "呜" * 200)]
    ex = build_multiturn_example(tok, turns, ctx=128)
    assert len(ex.input_ids) == len(ex.labels) <= 128
    # 截断后尾部仍是被监督的 Assistant 内容
    assert ex.labels[-1] != -100


def test_dataset_multiturn_chain_packing():
    from stateswap.data import S0Dataset
    from stateswap.tokenizer import WorldTokenizer

    tok = WorldTokenizer(VOCAB)
    data = [{"instruction": f"问题{i}", "output": f"回答{i}喵"} for i in range(9)]
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False)
        path = f.name
    ds = S0Dataset(tok, path, ctx=1024, turns=3)
    assert len(ds) == 3  # 9 对 → 3 条三轮链
    for ex in ds.examples:
        assert len(ex.input_ids) == len(ex.labels) <= 1024


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


def test_unique_4gram_ratio_detects_repetition():
    # 退化护栏的检测器：复读模板循环 → 低唯一率；正常文本 → 接近 1
    from stateswap.engine import _unique_4gram_ratio

    varied = "今天天气很好，我们一起去公园散步，聊了很多有趣的事情，还喝了咖啡。"
    repetitive = "主人...主人...(突然用爪子挠主人的手)号主人..." * 12
    assert _unique_4gram_ratio(varied) > 0.95
    assert _unique_4gram_ratio(repetitive) < 0.65
    assert _unique_4gram_ratio("短文本") == 1.0


def test_lowrank_roundtrip_and_rank():
    from stateswap.lowrank import lowrank_reconstruct, svd_truncate

    # 真实秩 r 的矩阵：rank-k ≥ r 截断应近乎无损，rank-k < r 误差 ≈ 尾部能量
    L, H = 2, 3
    a = torch.randn(L * H, 64, 10)
    s0 = (a @ a.transpose(1, 2)).reshape(L, H, 64, 64)  # 秩 ≤ 10
    rec = lowrank_reconstruct(svd_truncate(s0, 12))
    assert rec.shape == s0.shape
    assert torch.allclose(rec, s0, atol=1e-3)
    rec4 = lowrank_reconstruct(svd_truncate(s0, 4))
    err = float((rec4 - s0).norm() / s0.norm())
    assert 0.0 < err < 0.9  # 截断有损失但保留主方向


# ---------- persona factory ----------


def _write_card(tmp_path, **overrides):
    import json

    card = {
        "name": "test-neko",
        "description": "测试人格",
        "style_markers": ["喵"],
        "seed_dialogs": [{"instruction": "你好", "output": "喵～"}],
        "topics": ["闲聊"],
    }
    card.update(overrides)
    p = tmp_path / "card.json"
    p.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    return p


def test_load_card_defaults(tmp_path):
    from stateswap.factory import load_card

    card = load_card(_write_card(tmp_path))
    assert (card.n_pairs, card.turns) == (400, 2)
    assert (card.style_min, card.correct_min) == (0.75, 0.5)


def test_load_card_rejects_missing_and_bad_name(tmp_path):
    from stateswap.factory import load_card

    with pytest.raises(ValueError):
        load_card(_write_card(tmp_path, style_markers=[]))
    with pytest.raises(ValueError):
        load_card(_write_card(tmp_path, seed_dialogs=[{"instruction": "只有一半"}]))
    with pytest.raises(ValueError):
        load_card(_write_card(tmp_path, name="../路径穿越"))


def test_extract_json_array_variants():
    from stateswap.factory import extract_json_array

    items = [{"instruction": "a", "output": "b"}]
    assert extract_json_array('```json\n[{"instruction": "a", "output": "b"}]\n```') == items
    assert extract_json_array('好的，结果如下：\n[{"instruction": "a", "output": "b"}] 完毕') == items
    # 字符串内含 ] / \" 的括号配对边界
    tricky = '[{"instruction": "数组[1]结束]\\"", "output": "含\\"引号"}]'
    assert extract_json_array(tricky) == [
        {"instruction": '数组[1]结束]"', "output": '含"引号'}
    ]
    assert extract_json_array("没有任何 JSON") == []
    assert extract_json_array("[1, 2,") == []  # 截断


def test_clean_pair_normalization():
    from stateswap.factory import clean_pair

    assert clean_pair({"instruction": "User: 你好", "output": "Assistant: 喵～"}) == {
        "instruction": "你好",
        "output": "喵～",
    }
    assert clean_pair({"instruction": "x", "output": None}) is None
    assert clean_pair({"instruction": "a" * 500, "output": "b"}) is None  # 超长
    assert clean_pair(["not", "a", "dict"]) is None


def test_style_probes_and_scoring():
    from stateswap.factory import build_style_probes, neutral_correct, style_hit

    probes = build_style_probes(["Python", "Git", "SQL"], n=8)
    assert len(probes) == 8  # topic 不足 n 时循环补足，不越界
    assert any("Python" in p for p in probes)
    assert style_hit("喵呜～本喵来啦", ["喵", "本喵"])
    assert not style_hit("正常回答", ["喵"])
    assert neutral_correct("巴黎是法国的首都喵～", "巴黎")
    assert neutral_correct("H2O，也就是水", "h2o")  # 大小写不敏感
    assert not neutral_correct("不知道喵", "巴黎")


def test_llm_config_env_priority(monkeypatch):
    from stateswap import factory

    file_cfg = {
        "STATESWAP_LLM_BASE_URL": "https://from-file/v1",
        "STATESWAP_LLM_API_KEY": "sk-file",
        "STATESWAP_LLM_MODEL": "m-file",
    }
    monkeypatch.setattr(factory, "load_dotenv", lambda *a, **k: dict(file_cfg))
    for k in factory.ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    assert factory.llm_config()["model"] == "m-file"
    monkeypatch.setenv("STATESWAP_LLM_MODEL", "m-env")
    assert factory.llm_config()["model"] == "m-env"  # 环境变量优先于 .env
    monkeypatch.setattr(factory, "load_dotenv", lambda *a, **k: {})
    assert factory.llm_config() is None  # 三项不全 → 未配置


def test_load_dotenv_parsing(tmp_path):
    from stateswap.factory import load_dotenv

    f = tmp_path / "env"
    f.write_text(
        "# comment\nA = 'x y'\nB=\"z\"\nBAD LINE\nC=1\n", encoding="utf-8"
    )
    assert load_dotenv(f) == {"A": "x y", "B": "z", "C": "1"}
    assert load_dotenv(tmp_path / "nonexistent") == {}


def test_load_s0_payload_three_formats():
    from stateswap.lowrank import svd_truncate
    from stateswap.quant import quantize_int8
    from stateswap.s0 import load_s0_payload

    # svd_truncate 按 head_dim=64 硬编码 reshape，测试矩阵也用 64×64
    s0 = torch.randn(2, 3, 64, 64)
    assert torch.allclose(load_s0_payload({"s0": s0}), s0)
    # int8：逐 (L,H) 对称量化，误差 ≤ amax/254
    rec = load_s0_payload(quantize_int8(s0))
    assert rec.shape == s0.shape
    assert (rec - s0).abs().max() < 0.02
    # 低秩：构造真实秩 4 的矩阵，rank-4 截断近无损
    a = torch.randn(2 * 3, 64, 4)
    lowrank = (a @ a.transpose(1, 2)).reshape(2, 3, 64, 64)
    rec2 = load_s0_payload(svd_truncate(lowrank, 4))
    assert torch.allclose(rec2, lowrank, atol=1e-3)


# ---------- S0 editing ----------


def _orthonormal_pair(n, count, generator):
    """count 个相互正交的单位向量（QR）。"""
    a = torch.randn(n, count, generator=generator)
    q, _ = torch.linalg.qr(a)
    return q


def test_remove_subspace_kills_reference_component():
    from stateswap.editing import remove_subspace

    # 构造逐头 rank-1 的 N（u1v1ᵀ）与 Z（u2v2ᵀ），u2⊥u1、v2⊥v1：
    # 切除 ref=N 的 top-1 子空间应精确留下 Z
    g = torch.Generator().manual_seed(0)
    L, H, n = 2, 3, 64
    u = _orthonormal_pair(n, 2, g).reshape(1, 1, n, 2).expand(L, H, n, 2)
    v = _orthonormal_pair(n, 2, g).reshape(1, 1, n, 2).expand(L, H, n, 2)
    u1, u2 = u[..., 0], u[..., 1]
    v1, v2 = v[..., 0], v[..., 1]
    N = u1[..., None] @ v1[..., None, :]
    Z = u2[..., None] @ v2[..., None, :]
    out = remove_subspace(N + Z, N, k=1)
    assert torch.allclose(out, Z, atol=1e-4)


def test_remove_subspace_full_rank_zeroes():
    from stateswap.editing import remove_subspace

    g = torch.Generator().manual_seed(1)
    N = torch.randn(2, 2, 64, 64, generator=g)
    M = N + torch.randn(2, 2, 64, 64, generator=g)
    # k=64：ref 的奇异空间铺满全空间 → 切除后应为 0
    out = remove_subspace(M, N, k=64)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-3)


def test_inject_rank_and_scale():
    from stateswap.editing import inject, rank_k

    d = torch.randn(2, 2, 64, 64)
    out = inject(torch.zeros(2, 2, 64, 64), d, lam=2.0, k=3)
    assert torch.allclose(out, 2.0 * rank_k(d, 3), atol=1e-5)
    sv = torch.linalg.svdvals(out.reshape(4, 64, 64))
    assert torch.all(sv[:, 3:] < 1e-4)  # 注入部分秩 ≤ 3


def test_rank_k_truncation():
    from stateswap.editing import rank_k

    a = torch.randn(1, 1, 64, 64)
    assert torch.allclose(rank_k(a, 64), a, atol=1e-4)  # 全秩截断 = 原矩阵
    sv = torch.linalg.svdvals(rank_k(a, 2).reshape(1, 64, 64))[0]
    assert sv[2:].abs().max() < 1e-5


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

    def test_seed_session_invisible_init(self, engine):
        """吸引子种子：种子交换只写状态，不进 history/轮数。"""
        persona = "personas-legacy/neko-0.4b-v2/s0.pt"
        if not Path(persona).exists():
            pytest.skip("legacy 0.4B persona not present")
        engine.register_persona("seed-neko", persona)
        session = engine.new_session("seed-neko", seed="喵呜~早呀主人！")
        assert session.history == []
        assert session.turns == 0
        result = engine.chat(session.session_id, "在吗？", max_new_tokens=48, temperature=0.0)
        assert result["reply"] != ""

    def test_swap_keep_context_no_degeneration(self, engine):
        """回归：keep_context 换人格后 conv/ffn 必须重置——旧行为在对话中途
        swap 会产生 "AssAss…" 模板碎片（engineering-notes §7）。"""
        persona = "personas-legacy/neko-0.4b-v2/s0.pt"
        if not Path(persona).exists():
            pytest.skip("legacy 0.4B persona not present")
        engine.register_persona("swap-neko", persona)
        s = engine.new_session("swap-neko", context_replay=0)
        try:
            engine.chat(s.session_id, "早上好呀！", max_new_tokens=48, temperature=0.0)
            engine.chat(s.session_id, "今天的月亮又圆又亮。", max_new_tokens=48, temperature=0.0)
            engine.swap_persona(s.session_id, "none", keep_context=True)
            r1 = engine.chat(s.session_id, "早上好呀！", max_new_tokens=48, temperature=0.0)
            r2 = engine.chat(s.session_id, "讲个笑话", max_new_tokens=48, temperature=0.0)
            assert r1["reply"] and r2["reply"]
            from stateswap.engine import _unique_4gram_ratio

            assert _unique_4gram_ratio(r1["reply"]) > 0.5
        finally:
            engine.drop_session(s.session_id)

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
