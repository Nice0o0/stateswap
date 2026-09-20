# One-off: S₀ 空间相图实验（E1 序参量曲线 / E2 滞后回线 / E3 分层地形 /
# E4 几何-行为脱钩 / E5 温度稳健性）。不是包的一部分。
#
# 设计大纲（用户已确认）：neko-1.5b × zh2en-1.5b（边界修复后人格），
#   S(α) = α·neko + (1-α)·zh2en，贪心解码、独立会话、context_replay=0。
#   三个候选图景：H1 全程平滑共存（State Soup 的 Mamba 结论）/
#   H2 单点相变（第一轮 bug 数据图景）/ H3 双相变点夹共存区（相图）。
#
# 结果增量写入 docs/phase-transition.json；报告 docs/phase-transition.md。
import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

STYLE_PROMPTS = [
    "早上好呀！今天想吃小鱼干吗？",
    "陪我聊聊天吧，今天有点累",
    "你今天过得怎么样？",
    "我做了好多工作，好累啊",
    "周末我们去哪里玩？",
    "给我讲个笑话吧",
    "我有点想你了",
    "晚饭吃点什么好呢？",
]
ZH_SENTENCES = [
    "今天的月亮又圆又亮。",
    "孩子们在公园里放风筝。",
    "他因为堵车迟到了半个小时。",
    "这本书我已经读了三遍了。",
    "地铁上人太多了，我挤不上去。",
    "她用业余时间学会了弹吉他。",
    "台风来了，航班全部取消。",
    "程序员经常需要熬夜修 bug。",
]
STYLE_SHORT = STYLE_PROMPTS[:3]
ZH_SHORT = ZH_SENTENCES[:3]

COARSE_ALPHAS = [round(1.0 - 0.1 * i, 1) for i in range(11)]  # 1.0 -> 0.0

# E6 扩展探针集（E1 的 8 条 + 新 8 条），用于共存相逐 prompt 抖动分析
STYLE_PROMPTS_EXT = STYLE_PROMPTS + [
    "今天心情特别好，陪你聊聊天",
    "你在干嘛呢？",
    "给我唱首歌吧",
    "最近有什么开心的事吗？",
    "陪我看会儿星星吧",
    "你觉得我这个人怎么样？",
    "晚上想吃什么呢？",
    "抱抱",
]
ZH_SENTENCES_EXT = ZH_SENTENCES + [
    "办公室里很安静，只有键盘的声音。",
    "小雨一直在下，路面湿滑。",
    "他每天早上六点起床跑步。",
    "这家餐厅的招牌菜是红烧肉。",
    "高铁比飞机方便多了。",
    "她把房间收拾得干干净净。",
    "这部电影我看了两遍还想再看。",
    "冬天的早晨特别冷。",
]
COEXIST_ALPHAS = [0.65, 0.62, 0.6, 0.58, 0.55]  # 共存相及其边界
# E7 弛豫交换轮换池：每步相同的 settle/probe prompt 会诱发复读退化
# （状态雪崩，见 engineering-notes）——轮换不同 prompt 消除重复源
SETTLE_POOL = [
    "我们聊点别的吧", "换个话题怎么样", "你最近还好吗", "今天过得怎么样",
    "有什么想说的吗", "随便聊聊", "说点什么吧", "接下来想做什么",
    "我们接着聊", "再聊两句", "说说你的想法", "继续",
]


def is_english(reply: str) -> bool:
    return any(c.isascii() and c.isalpha() for c in reply) and not any(
        "\u4e00" <= c <= "\u9fff" for c in reply
    )


class Probe:
    def __init__(self, model_dir, vocab, neko="neko-1.5b", zh2en="zh2en-1.5b"):
        from stateswap.bench import NEKO_MARKERS
        from stateswap.engine import Engine

        self.markers = NEKO_MARKERS
        self.engine = Engine(model_dir, vocab)
        self.neko_name, self.zh2en_name = neko, zh2en
        engine = self.engine
        engine.register_persona(neko, ROOT / "personas" / neko / "s0.pt")
        engine.register_persona(zh2en, ROOT / "personas" / zh2en / "s0.pt")
        self.neko = engine.personas[neko].s0.cpu().clone()
        self.zh2en = engine.personas[zh2en].s0.cpu().clone()

    # ---- mixture management ----

    def register_mix(self, alpha: float) -> str:
        from stateswap.arithmetic import interpolate

        name = f"mix-{alpha:.2f}"
        if name not in self.engine.personas:
            self.engine.register_tensor(
                name, interpolate(self.neko, self.zh2en, alpha),
                {"mix": {"a": self.neko_name, "b": self.zh2en_name, "alpha": alpha}},
            )
        return name

    def mix_tensor(self, alpha: float) -> torch.Tensor:
        from stateswap.arithmetic import interpolate

        return interpolate(self.neko, self.zh2en, alpha)

    # ---- probing（persona= 新建独立会话；session_id= 在既有会话内连续探测） ----

    def _chat(self, session_id: str, text: str, max_new: int, temperature: float,
              top_p: float) -> dict:
        return self.engine.chat(session_id, text, max_new_tokens=max_new,
                                temperature=temperature, top_p=top_p,
                                rep_penalty=1.0, no_repeat_ngram=8)

    def style_probe(self, prompts=None, temperature: float = 0.0, max_new: int = 96,
                    persona: str | None = None, session_id: str | None = None) -> list[dict]:
        out = []
        for pr in (prompts or STYLE_PROMPTS):
            if session_id is None:
                sess = self.engine.new_session(persona, context_replay=0)
                try:
                    r = self._chat(sess.session_id, pr, max_new, temperature,
                                   1.0 if temperature == 0 else 0.8)
                finally:
                    self.engine.drop_session(sess.session_id)
            else:
                r = self._chat(session_id, pr, max_new, temperature,
                               1.0 if temperature == 0 else 0.8)
            out.append({"prompt": pr, "reply": r["reply"],
                        "hit": any(m in r["reply"] for m in self.markers),
                        "degenerated": r.get("degenerated", False)})
        return out

    def task_probe(self, sentences=None, temperature: float = 0.0, max_new: int = 64,
                   persona: str | None = None, session_id: str | None = None) -> list[dict]:
        out = []
        for s in (sentences or ZH_SENTENCES):
            if session_id is None:
                sess = self.engine.new_session(persona, context_replay=0)
                try:
                    r = self._chat(sess.session_id, s, max_new, temperature,
                                   1.0 if temperature == 0 else 0.8)
                finally:
                    self.engine.drop_session(sess.session_id)
            else:
                r = self._chat(session_id, s, max_new, temperature,
                               1.0 if temperature == 0 else 0.8)
            out.append({"zh": s, "reply": r["reply"], "english": is_english(r["reply"]),
                        "degenerated": r.get("degenerated", False)})
        return out

    @staticmethod
    def style_rate(samples): return sum(s["hit"] for s in samples) / max(1, len(samples))

    @staticmethod
    def task_rate(samples): return sum(s["english"] for s in samples) / max(1, len(samples))


def save(results: dict, out: Path):
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------- E1: order-parameter curve (coarse + adaptive refine) ----------------


def e1_curve(p: Probe, alphas: list[float]) -> list[dict]:
    points = []
    for a in alphas:
        name = p.register_mix(a)
        style = p.style_probe(persona=name)
        task = p.task_probe(persona=name)
        pts = {"alpha": a, "style_hit_rate": p.style_rate(style),
               "task_english_rate": p.task_rate(task),
               "style": style, "task": task,
               "sample_reply": style[0]["reply"][:120]}
        points.append(pts)
        print(f"E1 α={a:.2f}  style={pts['style_hit_rate']:.2f}  task={pts['task_english_rate']:.2f}")
    return points


def e1(p: Probe, results: dict, out: Path):
    t0 = time.time()
    coarse = e1_curve(p, COARSE_ALPHAS)
    results["e1"] = {"points": coarse}
    save(results, out)
    # 自适应加密：相邻点任一序参量跳变 ≥0.25 时插入中点（一轮，≤10 个）
    refine = []
    for a, b in zip(coarse, coarse[1:]):
        jump = max(abs(a["style_hit_rate"] - b["style_hit_rate"]),
                   abs(a["task_english_rate"] - b["task_english_rate"]))
        if jump >= 0.25:
            refine.append(round((a["alpha"] + b["alpha"]) / 2, 3))
    refine = sorted(set(refine))[:10]
    if refine:
        print(f"E1 refine: {refine}")
        fine = e1_curve(p, refine)
        results["e1"]["points"] = sorted(coarse + fine, key=lambda x: -x["alpha"])
        save(results, out)
    results["e1"]["seconds"] = round(time.time() - t0, 1)
    save(results, out)


# ---------------- E4: geometry vs behavior decoupling ----------------


def e4(p: Probe, results: dict, out: Path):
    rows = []
    for pt in results.get("e1", {}).get("points", []):
        a = pt["alpha"]
        m = p.mix_tensor(a).float()
        row = {"alpha": a,
               "fro_dist_neko": round(float((m - p.neko).norm()), 4),
               "fro_dist_zh2en": round(float((m - p.zh2en).norm()), 4),
               "norm": round(float(m.norm()), 4)}
        for label, ref in (("cos_vs_neko", p.neko), ("cos_vs_zh2en", p.zh2en)):
            per_layer = []
            for i in range(m.shape[0]):
                x = m[i].reshape(m.shape[1], -1)
                y = ref[i].reshape(ref.shape[1], -1)
                per_layer.append(round(torch.nn.functional.cosine_similarity(
                    x, y, dim=-1).mean().item(), 4))
            row[label] = per_layer
        rows.append(row)
    results["e4"] = {"points": rows}
    save(results, out)
    print(f"E4 done: {len(rows)} points (pure tensor math)")


# ---------------- E2: hysteresis / in-session dynamics ----------------


def blend_running_state(p: Probe, session, target: torch.Tensor, beta: float = 0.35):
    """把会话的运行态递归状态向目标 S₀ 部分混合（保留累计 token 内容 → 路径依赖）。"""
    eng = p.engine
    with torch.no_grad():
        for i in range(eng.model.config.num_hidden_layers):
            cur = session.cache[i]["recurrent_state"]
            tgt = target[i].to(cur.device, cur.dtype).unsqueeze(0)
            session.cache.update(recurrent_state=(1 - beta) * cur + beta * tgt,
                                 layer_idx=i, offset=0)


def e2(p: Probe, results: dict, out: Path):
    eng = p.engine
    ladder = COARSE_ALPHAS  # 1.0 -> 0.0
    res: dict = {}

    # E2a 换人格阶梯（keep_context=True）：正/反两方向
    for tag, seq in (("down", ladder), ("up", list(reversed(ladder)))):
        sess = eng.new_session(p.register_mix(seq[0]), context_replay=0)
        rows = []
        for a in seq:
            eng.swap_persona(sess.session_id, p.register_mix(a), keep_context=True)
            s = p.style_probe(prompts=STYLE_SHORT[:2], session_id=sess.session_id)
            t = p.task_probe(sentences=ZH_SHORT[:2], session_id=sess.session_id)
            rows.append({"alpha": a, "style": p.style_rate(s), "task": p.task_rate(t)})
        eng.drop_session(sess.session_id)
        res[f"swap_{tag}"] = rows
        print(f"E2a swap-{tag}: " + " ".join(
            f"{r['alpha']:.1f}:s{r['style']:.1f}/t{r['task']:.1f}" for r in rows))

    # E2b 会话内弛豫：同一探针连发 3 轮（行为是否随状态演化翻转）
    relax = {}
    for a in (1.0, 0.7, 0.5, 0.3, 0.0):
        sess = eng.new_session(p.register_mix(a), context_replay=0)
        turns = []
        for t in range(3):
            s = p.style_probe(prompts=STYLE_SHORT, session_id=sess.session_id)
            turns.append(p.style_rate(s))
        eng.drop_session(sess.session_id)
        relax[f"{a:.1f}"] = turns
        print(f"E2b relax α={a:.1f}: {turns}")
    res["relaxation"] = relax

    # E2c 对照：纯人格会话里同一探针连发 5 轮（自然漂移基线）
    ctrl = {}
    for name, prompts in ((p.neko_name, STYLE_SHORT), (p.zh2en_name, ZH_SHORT)):
        sess = eng.new_session(name, context_replay=0)
        turns = []
        for _ in range(5):
            if name == p.neko_name:
                turns.append(p.style_rate(p.style_probe(prompts=prompts, session_id=sess.session_id)))
            else:
                turns.append(p.task_rate(p.task_probe(sentences=prompts, session_id=sess.session_id)))
        eng.drop_session(sess.session_id)
        ctrl[name] = turns
        print(f"E2c control {name}: {turns}")
    res["control"] = ctrl

    # E2d 渐变混合行走（真·路径依赖）：运行态逐步向目标混合，正/反两方向
    for tag, seq in (("down", ladder), ("up", list(reversed(ladder)))):
        start = p.register_mix(seq[0])
        sess = eng.new_session(start, context_replay=0)
        rows = []
        for a in seq:
            if a != seq[0]:
                blend_running_state(p, sess, p.mix_tensor(a))
            s = p.style_probe(prompts=STYLE_SHORT[:2], session_id=sess.session_id)
            t = p.task_probe(sentences=ZH_SHORT[:2], session_id=sess.session_id)
            rows.append({"alpha": a, "style": p.style_rate(s), "task": p.task_rate(t)})
        eng.drop_session(sess.session_id)
        res[f"blend_{tag}"] = rows
        print(f"E2d blend-{tag}: " + " ".join(
            f"{r['alpha']:.1f}:s{r['style']:.1f}/t{r['task']:.1f}" for r in rows))

    results["e2"] = res
    save(results, out)


# ---------------- E3: layer-splice topography ----------------


def e3(p: Probe, results: dict, out: Path):
    L = p.neko.shape[0]
    rows = {"front_neko": [], "front_zh2en": []}
    for tag, front, back in (("front_neko", p.neko, p.zh2en),
                             ("front_zh2en", p.zh2en, p.neko)):
        for k in range(L + 1):
            tensor = torch.cat([front[:k], back[k:]], dim=0)
            name = f"splice-{tag}-{k:02d}"
            p.engine.register_tensor(name, tensor, {"splice": {"front": tag, "k": k}})
            s = p.style_probe(persona=name, prompts=STYLE_SHORT)
            t = p.task_probe(persona=name, sentences=ZH_SHORT)
            rows[tag].append({"k": k, "style": p.style_rate(s), "task": p.task_rate(t)})
            p.engine.delete_persona(name)
        print(f"E3 {tag}: " + " ".join(
            f"k{r['k']}:{r['style']:.1f}/{r['task']:.1f}" for r in rows[tag]))
    results["e3"] = rows
    save(results, out)


# ---------------- E5: temperature robustness at transition points ----------------


def e5(p: Probe, results: dict, out: Path):
    pts = sorted(results.get("e1", {}).get("points", []), key=lambda x: -x["alpha"])
    # 找风格序参量首次跌破 0.5 的 α（从 neko 端向 zh2en 端）
    t_alpha = next((a["alpha"] for a in pts if a["style_hit_rate"] < 0.5), None)
    if t_alpha is None:
        results["e5"] = {"skipped": "no style transition found in E1"}
        save(results, out)
        return
    rows = []
    for a in {round(max(0.0, t_alpha - 0.05), 3), round(min(1.0, t_alpha + 0.05), 3)}:
        name = p.register_mix(a)
        for temp in (0.0, 0.7):
            style = p.style_probe(persona=name, temperature=temp)
            task = p.task_probe(persona=name, temperature=temp)
            rows.append({"alpha": a, "temperature": temp,
                         "style_hit_rate": p.style_rate(style),
                         "task_english_rate": p.task_rate(task)})
            print(f"E5 α={a:.2f} T={temp}: style={rows[-1]['style_hit_rate']:.2f} "
                  f"task={rows[-1]['task_english_rate']:.2f}")
    results["e5"] = {"transition_alpha": t_alpha, "points": rows}
    save(results, out)


def _classify(reply: str, markers) -> str:
    """回复三分类：风格命中（标记词）/ 任务（纯英文）/ 其他（如中文无标记）。"""
    if any(m in reply for m in markers):
        return "style"
    if is_english(reply):
        return "task"
    return "other"


def e6(p: Probe, results: dict, out: Path):
    """共存相逐 prompt 抖动：16+16 大探针集，逐 prompt 三分类（风格/任务/其他），
    回答"共存相是逐样本吸引子选择还是混合风格回复"。"""
    rows = []
    for a in COEXIST_ALPHAS:
        name = p.register_mix(a)
        style = p.style_probe(persona=name, prompts=STYLE_PROMPTS_EXT)
        task = p.task_probe(persona=name, sentences=ZH_SENTENCES_EXT)
        row = {
            "alpha": a,
            "style_class": [_classify(s["reply"], p.markers) for s in style],
            "task_class": [_classify(t["reply"], p.markers) for t in task],
            "style": style, "task": task,
        }
        row["style_hits"] = row["style_class"].count("style")
        row["task_hits"] = row["task_class"].count("task")
        row["other"] = row["style_class"].count("other") + row["task_class"].count("other")
        rows.append(row)
        print(f"E6 α={a:.2f} style {row['style_hits']}/16 task {row['task_hits']}/16 "
              f"other {row['other']}/32")
    results["e6"] = {"alphas": COEXIST_ALPHAS, "points": rows}
    save(results, out)


def e7(p: Probe, results: dict, out: Path):
    """平衡态回扫：每步混合（两次 β=0.5 → ~75% 收敛）后经 2 轮中性交换弛豫，
    再读序参量。与 E2d 的区别：E2d 混合后立即测量（测的是状态惯性），
    本协议让动力学在每步安定后读数——平衡态滞后回线的正确测法。"""
    eng = p.engine
    res = {}
    for tag, seq in (("down", COARSE_ALPHAS), ("up", list(reversed(COARSE_ALPHAS)))):
        sess = eng.new_session(p.register_mix(seq[0]), context_replay=0)
        rows = []
        for i, a in enumerate(seq):
            if i > 0:
                target = p.mix_tensor(a)
                blend_running_state(p, sess, target, beta=0.5)
                blend_running_state(p, sess, target, beta=0.5)
            settle = []
            for j in range(2):
                q = SETTLE_POOL[(i * 2 + j) % len(SETTLE_POOL)]
                r = eng.chat(sess.session_id, q, max_new_tokens=48,
                             temperature=0.0, top_p=1.0, rep_penalty=1.0, no_repeat_ngram=8)
                settle.append(f"{q}:{r['reply'][:30]}")
            s = p.style_probe(prompts=STYLE_SHORT, session_id=sess.session_id)
            t = p.task_probe(sentences=ZH_SHORT, session_id=sess.session_id)
            rows.append({"alpha": a, "style": p.style_rate(s), "task": p.task_rate(t),
                         "settle": settle})
            print(f"E7 {tag} α={a:.1f} style={rows[-1]['style']:.2f} "
                  f"task={rows[-1]['task']:.2f} settle0={settle[0][:18]!r}")
        eng.drop_session(sess.session_id)
        res[tag] = rows
    results["e7"] = res
    save(results, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "rwkv7-1.5b-world-hf"))
    ap.add_argument("--vocab", default=str(ROOT / "vendor" / "rwkv_vocab_v20230424.txt"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "phase-transition.json"))
    ap.add_argument("--stages", default="e1,e4,e2,e3,e5")
    ap.add_argument("--smoke", action="store_true",
                    help="E1 only, 3 alphas — script sanity check")
    args = ap.parse_args()

    p = Probe(args.model, args.vocab)
    out = Path(args.out)
    results = {}
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
    results.setdefault("meta", {
        "model": args.model, "neko": p.neko_name, "zh2en": p.zh2en_name,
        "protocol": "greedy, fresh sessions, context_replay=0, rep_penalty=1.0, ngram=8",
        "decode": "temperature=0 top_p=1.0 (except e5)",
    })
    save(results, out)

    stages = args.stages.split(",")
    if args.smoke:
        results["e1"] = {"points": e1_curve(p, [1.0, 0.5, 0.0])}
        save(results, out)
        return
    for s in stages:
        s = s.strip()
        print(f"===== stage {s} =====")
        if s == "e1":
            e1(p, results, out)
        elif s == "e4":
            e4(p, results, out)
        elif s == "e2":
            e2(p, results, out)
        elif s == "e3":
            e3(p, results, out)
        elif s == "e5":
            e5(p, results, out)
        elif s == "e6":
            e6(p, results, out)
        elif s == "e7":
            e7(p, results, out)
        else:
            raise SystemExit(f"unknown stage {s}")
    print("all stages done")


if __name__ == "__main__":
    main()
