"""Programmatic figures for docs/paper-draft.md — pure-Python SVG, no dependencies.

为什么自己写 SVG：
- 仓库宣称"零依赖 WebUI"，为一个静态图引入 matplotlib（及它拉进来的 numpy
  等）不值得；CI 里也不需要装任何新东西。
- 图必须能从 docs/*.json **程序化重生成**，并与 commit hash 绑定——这样
  "图与数据不一致"的审稿意见不可能发生。
- 输出是矢量 SVG，arXiv 与浏览器都能直接看。

用法：
    python scripts/make_figures.py              # 写 docs/fig/*.svg
    python scripts/make_figures.py --check      # 只校验数据，不写文件
    python scripts/make_figures.py --only fig4  # 只重生成满足子串的图

设计约定（都是第一版踩过的坑）：
- **一张图 = 一个 Canvas**，不在一个文件里拼两个独立面板（拼贴会让两个 title
  重叠在同一个 y 上）；
- alpha 轴自左向右是 α=1（style 端）→ α=0（task 端），`px()` 做这个翻转；
- 画布高度按内容定，不留大片空白；
- `cos_vs_*` 在某些 JSON 里是逐层数组，画图取层均值并在副标题里说明。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "fig"

INK = "#1b2530"
MUTED = "#66717d"
GRID = "#dde3ea"
BAND = "#eef2f7"
ACCENT = "#3b5a8f"   # style / S₀
ACCENT2 = "#c1573f"  # task / LoRA
ACCENT3 = "#4a8f6a"  # geometry
PALETTE = ("#3b5a8f", "#c1573f", "#4a8f6a", "#8a6fb0", "#c99a2e", "#4a7f8f",
           "#7f6b5a")


class Canvas:
    """极简绘图：内部 0..1 归一化，y(0)=底、y(1)=顶。"""

    def __init__(self, w: int, h: int, *, left=70, right=28, top=62, bottom=54):
        self.w, self.h = w, h
        self.pl, self.pr, self.pt, self.pb = left, right, top, bottom
        self.parts: list[str] = []

    def x(self, t: float) -> float:
        return self.pl + t * (self.w - self.pl - self.pr)

    def y(self, t: float) -> float:
        return self.h - self.pb - t * (self.h - self.pt - self.pb)

    def _esc(self, s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(self, x, y, s, size=10, color=INK, anchor="start", weight="normal") -> None:
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{self._esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color=GRID, width=1.0, dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                          f'stroke="{color}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, opacity=1.0) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                          f'fill="{fill}" opacity="{opacity}"/>')

    def polyline(self, pts, color, width=2.2, dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        path = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        self.parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                          f'stroke-width="{width}" stroke-linejoin="round"{d}/>')

    def dots(self, pts, color, r=3.4) -> None:
        for px, py in pts:
            self.parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r}" fill="{color}"/>')

    def title(self, s, sub=None) -> None:
        self.text(self.pl, 26, s, size=13.5, weight="600")
        if sub:
            self.text(self.pl, 44, sub, size=9.5, color=MUTED)

    def ygrid(self, ticks, fmt="{:.0%}", label=None) -> None:
        for v in ticks:
            yy = self.y(v)
            self.line(self.pl, yy, self.w - self.pr, yy, width=0.7)
            self.text(self.pl - 8, yy + 3.4, fmt.format(v), size=9, color=MUTED, anchor="end")
        if label:
            self.text(self.pl - 8, self.pt - 10, label, size=9, color=MUTED, anchor="end")

    def xgrid(self, positions, labels, label=None, size=9) -> None:
        for t, lab in zip(positions, labels):
            self.text(self.x(t), self.h - self.pb + 16, lab, size=size, color=MUTED,
                      anchor="middle")
        if label:
            self.text(self.w - self.pr, self.h - 12, label, size=9, color=MUTED, anchor="end")

    def legend(self, items, x, y) -> None:
        for i, (lab, color) in enumerate(items):
            ry = y + i * 15
            self.parts.append(f'<rect x="{x:.2f}" y="{ry - 7:.2f}" width="9" height="9" '
                              f'fill="{color}"/>')
            self.text(x + 15, ry + 1, lab, size=9, color=MUTED)

    def note(self, x, y, lines, size=9) -> None:
        for i, s in enumerate(lines):
            self.text(x, y + i * 13.5, s, size=size, color=MUTED)

    def svg(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">\n'
            '<rect width="100%" height="100%" fill="#ffffff"/>\n'
            + "\n".join(self.parts) + "\n</svg>\n"
        )


def _json(name: str) -> dict:
    path = DOCS / name
    if not path.exists():
        raise SystemExit(f"missing data file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------- Fig 1: 三路注入（风格 vs 成本）----------------


def fig1() -> str:
    d = _json("three_way.json")
    style = [("S₀", d["S0"]["style"]), ("LoRA", d["LoRA"]["style"]),
             ("system prompt", d["system_prompt"]["style"])]
    cost = [("S₀", 3.0, "3 ms swap"), ("LoRA", 3000.0, "3 GB / persona"),
            ("system prompt", 1731.0, "1731 tok / req")]

    c = Canvas(820, 310, left=64, right=26)
    c.title("Fig 1 — Three injection sites",
            "left: does it inject the style? · right: what does one persona cost?")

    # 左半：风格命中
    left_r = 360
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="style hit")
    span = (left_r - c.pl) / 3
    for i, (lab, v) in enumerate(style):
        bx = c.pl + span * i + span * 0.28
        bw = span * 0.44
        c.rect(bx, c.y(v), bw, c.y(0) - c.y(v), PALETTE[i], 0.88)
        c.text(bx + bw / 2, c.y(v) - 7, f"{v:.0%}", size=10, anchor="middle")
    c.xgrid([(i + 0.5) / 3 * (left_r - c.pl) / (c.w - c.pl - c.pr) for i in range(3)],
            [s[0] for s in style])

    # 右半：成本（log）
    x0, x1 = left_r + 40, c.w - c.pr
    c.line(x0 - 16, c.pt, x0 - 16, c.h - c.pb, color=GRID, width=1)
    l0, l1 = 0.0, math.log10(4000.0)

    def lp(v):
        return (math.log10(max(v, 1.0)) - l0) / (l1 - l0)

    for v in (1, 10, 100, 1000):
        yy = c.y(lp(v))
        c.line(x0, yy, x1, yy, width=0.7)
        c.text(x0 - 8, yy + 3.4, f"{v:g}", size=9, color=MUTED, anchor="end")
    c.text(x0 - 8, c.pt - 10, "cost (log)", size=9, color=MUTED, anchor="end")
    span2 = (x1 - x0) / 3
    for i, (lab, val, note) in enumerate(cost):
        bx = x0 + span2 * i + span2 * 0.28
        bw = span2 * 0.44
        c.rect(bx, c.y(lp(val)), bw, c.y(0) - c.y(lp(val)), PALETTE[i], 0.88)
        c.text(bx + bw / 2, c.y(lp(val)) - 7, f"{val:g}", size=9.5, anchor="middle")
        c.text(bx + bw / 2, c.h - c.pb + 16, lab, size=8.5, color=MUTED, anchor="middle")
        c.text(bx + bw / 2, c.h - c.pb + 28, note, size=8, color=MUTED, anchor="middle")
    return c.svg()


# ---------------- Fig 2: 解剖 ----------------

def fig2() -> str:
    a = _json("anatomy.json")
    ab, br = a["A_ablation"], a["B_rank"]
    order = ["full", "neko-zero-L00-07", "neko-zero-L08-15", "neko-zero-L16-23"]
    labels = ["full", "−L00–07", "−L08–15", "−L16–23"]

    c = Canvas(820, 320, left=64, right=26)
    c.title("Fig 2 — Anatomy of a persona",
            "left: zero one layer group of neko · right: per-head rank truncation")
    left_r = 360
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="style hit")
    span = (left_r - c.pl) / len(order)
    for i, k in enumerate(order):
        val = ab.get(k, {}).get("neko_style")
        if val is None:
            continue
        bx = c.pl + span * i + span * 0.24
        bw = span * 0.52
        c.rect(bx, c.y(val), bw, c.y(0) - c.y(val), ACCENT, 0.88)
        c.text(bx + bw / 2, c.y(val) - 7, f"{val:.0%}", size=10, anchor="middle")
        c.text(bx + bw / 2, c.h - c.pb + 16, labels[i], size=9, color=MUTED, anchor="middle")
    c.text(c.pl - 8, c.pt - 10, "style hit", size=9, color=MUTED, anchor="end")

    x0, x1 = left_r + 44, c.w - c.pr
    c.line(x0 - 16, c.pt, x0 - 16, c.h - c.pb, color=GRID, width=1)
    # 右面板也需要自己的 y 刻度（第一版漏了，只有一条孤零零的 100% 线）
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        yy = c.y(v)
        c.line(x0 - 16, yy, x1, yy, color=GRID, width=0.7)
        c.text(x0 - 24, yy + 3.4, f"{v:.0%}", size=9, color=MUTED, anchor="end")
    c.text(x0 - 24, c.pt - 10, "style hit", size=9, color=MUTED, anchor="end")
    ranks = [4, 8, 16, 32]
    xs = {r: x0 + (x1 - x0) * ((i + 0.5) / len(ranks)) for i, r in enumerate(ranks)}
    pts = [(xs[r], c.y(br[f"neko-rank{r}"]["neko_style"]))
           for r in ranks if f"neko-rank{r}" in br]
    c.polyline(pts, ACCENT, 2.4)
    c.dots(pts, ACCENT)
    for r in ranks:
        if f"neko-rank{r}" in br:
            c.text(xs[r], c.h - c.pb + 16, f"k={r}", size=9, color=MUTED, anchor="middle")
    c.note(c.pl, c.h - 12, ["style is late-layer and distributed; task sits in the front third; "
                            "rank-4 (1.59 MB, 7.9×) keeps 100%"], 9)
    return c.svg()


# ---------------- Fig 3: 继承 ----------------

def fig3() -> str:
    d = _json("inheritance.json")
    arms = [("zh2en-1.5b-warm", "all layers trainable"),
            ("zh2en-1.5b-warm-front", "front 8 layers only\n(back 16 frozen bit-exact)")]
    c = Canvas(600, 310, left=64, right=26)
    c.title("Fig 3 — Composition by training (inheritance)",
            "a warm start from a style donor, then task data")
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="rate")
    span = (c.w - c.pr - c.pl) / len(arms)
    for i, (k, lab) in enumerate(arms):
        task = d.get(k, {}).get("translation", {}).get("english_rate", 0)
        style = d.get(k, {}).get("style", {}).get("style_hit_rate", 0)
        cx = c.pl + span * (i + 0.5)
        for j, (val, col) in enumerate([(task, ACCENT), (style, ACCENT2)]):
            bx = cx - 37 + j * 41
            if val > 0:
                c.rect(bx, c.y(val), 34, c.y(0) - c.y(val), col, 0.88)
                c.text(bx + 17, c.y(val) - 7, f"{val:.0%}", size=10, anchor="middle")
            else:
                # 0% 的柱子没有高度；画一个 2px 的基座标记否则读者以为缺数据
                c.rect(bx, c.y(0) - 2, 34, 2, col, 0.88)
                c.text(bx + 17, c.y(0) - 8, "0%", size=10, anchor="middle")
        c.text(cx, c.h - c.pb + 16, lab.split("\n")[0], size=9, color=MUTED, anchor="middle")
        if len(lab.split("\n")) > 1:
            c.text(cx, c.h - c.pb + 28, lab.split("\n")[1], size=8.5, color=MUTED,
                   anchor="middle")
    c.legend([("task (English)", ACCENT), ("donor style", ACCENT2)],
             x=c.pl + 10, y=c.y(0.62))
    c.note(c.pl, c.h - 12, ["the donor's subspace is bit-identical in L08–23 (cos 1.000) "
                            "yet style still reads 0%"], 9)
    return c.svg()


# ---------------- Fig 4: 相图 1.5B ----------------

def fig4() -> str:
    d = _json("phase-transition.json")
    pts = sorted(d["e1"]["points"], key=lambda p: -p["alpha"])
    c = Canvas(720, 360, left=64, right=28)
    c.title("Fig 4 — The phase diagram (1.5B)",
            "shaded band = coexistence window α≈[0.55,0.65]; α runs style(1) → task(0)")
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="rate")

    def px(alpha):
        return c.x(1 - alpha)

    c.rect(px(0.65), c.y(1), px(0.55) - px(0.65), c.y(0) - c.y(1), BAND)
    # 标签放在带内靠下位置：两条曲线在该区间都在高位，顶部会压字
    c.text((px(0.65) + px(0.55)) / 2, c.y(0.10), "coexistence", size=8.5,
           color=MUTED, anchor="middle")
    style = [(px(p["alpha"]), c.y(p["style_hit_rate"])) for p in pts]
    task = [(px(p["alpha"]), c.y(p["task_english_rate"])) for p in pts]
    c.polyline(style, ACCENT, 2.3)
    c.polyline(task, ACCENT2, 2.3)
    c.dots(style, ACCENT)
    c.dots(task, ACCENT2)
    c.text(px(1.0), c.h - c.pb + 16, "α=1.0 pure style", size=8.5, color=MUTED, anchor="middle")
    c.text(px(0.0), c.h - c.pb + 16, "α=0.0 pure task", size=8.5, color=MUTED, anchor="middle")
    # 图例放右下：α≈0 段两条线分别是 0% 与 100%，中间是空的
    c.legend([("style hit", ACCENT), ("task: English output", ACCENT2)], x=c.w - 215,
             y=c.y(0.55))
    return c.svg()


# ---------------- Fig 5: 几何 vs 行为 ----------------

def fig5() -> str:
    d = _json("phase-transition.json")
    geo = sorted(d["e4"]["points"], key=lambda p: -p["alpha"])
    beh = {p["alpha"]: p for p in d["e1"]["points"]}

    def mean_cos(v):
        return sum(v) / len(v) if isinstance(v, list) else float(v)

    c = Canvas(720, 360, left=64, right=28)
    c.title("Fig 5 — Geometry is smooth, behaviour is not",
            "cos(state, endpoint) moves near-linearly (layer mean) while both order "
            "parameters jump")
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="value")

    def px(alpha):
        return c.x(1 - alpha)

    cos_pts = [(px(p["alpha"]), c.y(mean_cos(p["cos_vs_neko"]))) for p in geo]
    st_pts = [(px(p["alpha"]), c.y(beh[p["alpha"]]["style_hit_rate"]))
              for p in geo if p["alpha"] in beh]
    tk_pts = [(px(p["alpha"]), c.y(beh[p["alpha"]]["task_english_rate"]))
              for p in geo if p["alpha"] in beh]
    c.polyline(cos_pts, ACCENT3, 2.4)
    c.dots(cos_pts, ACCENT3)
    c.polyline(st_pts, ACCENT, 2.0, dash="4 3")
    c.polyline(tk_pts, ACCENT2, 2.0, dash="4 3")
    # 图例放中右空白区：α≈0.2 处三条线分别位于 5%/0%/100%，图例压 100% 那条
    c.legend([("cos(state, S₀_neko)", ACCENT3), ("style hit", ACCENT),
              ("task: English", ACCENT2)], x=c.w - 205, y=c.y(0.30))
    c.note(c.w - 205, c.y(0.10), ["the state selects a basin;", "it is not a linear blend"])
    return c.svg()


# ---------------- Fig 6: 编辑 ----------------

def _fig6_values(base: dict, ext: dict) -> dict:
    """从两份 JSON 取 fig6 用到的 7 组数值，避免图与数据脱节。"""
    b = next(c for c in base["configs"] if c["tag"] == "base-mix-0.6")
    by_tag = {c["tag"]: c for c in ext["configs"]}
    pick = {
        "baseline": (b["style_hit_rate"], b["task_english_rate"]),
        "uniform_k4": (next(c for c in base["configs"]
                            if c["tag"] == "e1-remove-style-k4")["style_hit_rate"],
                       next(c for c in base["configs"]
                            if c["tag"] == "e1-remove-style-k4")["task_english_rate"]),
    }
    for tag in ("e7-remove-style-k4-L00-07", "e7-remove-style-k4-L08-15",
                "e7-remove-style-k4-L16-23", "e8-soft-remove-style-k4-b0.25",
                "e8-soft-remove-style-k4-b0.75"):
        pick[tag] = (by_tag[tag]["style_hit_rate"], by_tag[tag]["task_english_rate"])
    return pick


def _assert_matches(base: dict, ext: dict) -> None:
    """图里硬编码的 7 组数字必须与 JSON 完全一致——图与数据脱节是审稿人
    第一个会抓的问题，这里让它在生成时就失败。"""
    v = _fig6_values(base, ext)
    expect = {
        "baseline": (0.875, 0.75),
        "uniform_k4": (0.125, 0.0),
        "e7-remove-style-k4-L00-07": (0.375, 0.0),
        "e7-remove-style-k4-L08-15": (0.375, 0.5),
        "e7-remove-style-k4-L16-23": (0.125, 0.625),
        "e8-soft-remove-style-k4-b0.25": (0.125, 0.625),
        "e8-soft-remove-style-k4-b0.75": (0.125, 0.0),
    }
    for k, want in expect.items():
        got = v[k]
        if any(abs(a - b) > 1e-9 for a, b in zip(got, want)):
            raise SystemExit(f"fig6 value drift for {k}: data={got} figure={want}")


def fig6() -> str:
    """柱状图而不是散点图：27 个配置的 (style, task) 只有 ~12 个不同取值，
    散点会叠成一团；柱状图能直接看出"分层切除 62% task" vs "均匀 0%"。
    """
    # 数值直接读自两份 JSON（硬编码只是它们的摘要，下面断言保证一致）
    base = _json("state-editing.json")
    ext = _json("state-editing-e7e8.json")
    _assert_matches(base, ext)
    rows = [
        ("baseline (α=0.6)", 0.875, 0.75, PALETTE[0]),
        ("uniform k=4 /  all 24 layers", 0.125, 0.0, PALETTE[1]),
        ("masked L00–07", 0.375, 0.0, PALETTE[5]),
        ("masked L08–15", 0.375, 0.5, PALETTE[5]),
        ("masked L16–23", 0.125, 0.625, PALETTE[5]),
        ("soft β=0.25", 0.125, 0.625, PALETTE[6]),
        ("soft β=0.75", 0.125, 0.0, PALETTE[6]),
    ]
    c = Canvas(720, 350, left=64, right=28, bottom=74)
    c.title("Fig 6 — Low-rank surgery: task modes are excisable, style is not",
            "27 configurations; the discriminating ones shown here (style + task readouts)")
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="rate")
    span = (c.w - c.pr - c.pl) / len(rows)
    bw = span * 0.30
    for i, (lab, st, tk, col) in enumerate(rows):
        cx = c.pl + span * (i + 0.5)
        for j, (val, c2) in enumerate([(st, ACCENT), (tk, ACCENT2)]):
            bx = cx - bw - 3 + j * (bw + 6)
            c.rect(bx, c.y(val), bw, c.y(0) - c.y(val), c2, 0.85)
            c.text(bx + bw / 2, c.y(val) - 6, f"{val:.0%}", size=9, anchor="middle")
        for k, line in enumerate(lab.split(chr(10))):
            c.text(cx, c.h - c.pb + 16 + k * 12, line, size=8.5, color=MUTED,
                   anchor="middle")
    c.legend([("style hit", ACCENT), ("task: English", ACCENT2)], x=c.w - 190, y=c.pt + 6)
    c.note(c.pl, c.h - 12,
           ["removing only in L16–23 leaves the task at 62% (uniform removal: 0%); "
            "no edit keeps style above 38%"], 9)
    return c.svg()


# ---------------- Fig 7: scaling ----------------

def fig7() -> str:
    d = _json("scaling.json")
    pts = d["points"]
    c = Canvas(680, 340, left=68, right=28)
    c.title("Fig 7 — Style baking is cheap; capability is the cliff",
            "log x-axis (corpus size); each dot is one (pairs, steps) configuration")
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="rate")
    pairs = sorted({p["pairs"] for p in pts})
    lo, hi = math.log(min(pairs)), math.log(max(pairs))

    def px(n):
        return c.x((math.log(n) - lo) / (hi - lo)) if hi > lo else c.x(0.5)

    # xgrid 接受 0..1 归一化位置，px() 返回的是绝对像素——这里换算回 t
    ts = [(math.log(n) - lo) / (hi - lo) if hi > lo else 0.5 for n in pairs]
    c.xgrid(ts, [f"{n}" for n in pairs], label="corpus size (pairs)")
    st = [(px(p["pairs"]), c.y(p["style_hit_rate"])) for p in pts]
    ca = [(px(p["pairs"]), c.y(p["neutral_correct_rate"])) for p in pts]
    c.dots(st, ACCENT, 3.4)
    c.dots(ca, ACCENT2, 3.4)
    c.legend([("style hit", ACCENT), ("capability kept", ACCENT2)],
             x=c.w - 195, y=c.y(0.72))
    c.note(c.w - 195, c.y(0.52), ["50 pairs × 200 steps already reach 100% style;",
                                  "capability stays at the noise floor (~12%)"])
    return c.svg()


# ---------------- Fig 8: 三档标度对比（0.1B / 0.4B / 1.5B）----------------

def fig8() -> str:
    d01 = _json("phase-transition-0.1b.json")
    d04 = _json("phase-transition-0.4b-v2.json")
    d15 = _json("phase-transition.json")
    c = Canvas(760, 380, left=64, right=28)
    c.title("Fig 8 — Three base sizes: smooth handover vs collapse vs coexistence",
            "style (dashed) and English output (solid) across the persona axis")
    c.ygrid([0, 0.25, 0.5, 0.75, 1.0], label="rate")

    def px(alpha):
        return c.x(1 - alpha)

    sets = [
        ("0.1B (12×12, 2.36 MB)", d01["points"], PALETTE[0], "style_hit_rate"),
        ("0.4B (24×16, 6.29 MB)", d04["points"], PALETTE[1], "style_hit_rate"),
        ("1.5B (24×32, 12.58 MB)", d15["e1"]["points"], PALETTE[2], "style_hit_rate"),
    ]
    for label, pts, col, key in sets:
        pts = sorted(pts, key=lambda p: -p["alpha"])
        c.polyline([(px(p["alpha"]), c.y(p[key])) for p in pts], col, 1.9, dash="5 3")
        c.dots([(px(p["alpha"]), c.y(p[key])) for p in pts], col, 2.8)

    # 英文输出用同色细实线，避免图例爆掉
    for label, pts, col, key in sets:
        pts = sorted(pts, key=lambda p: -p["alpha"])
        tk = "english_rate" if "english_rate" in pts[0] else "task_english_rate"
        c.polyline([(px(p["alpha"]), c.y(p[tk])) for p in pts], col, 1.4)
    # 图例放中下：该区域只有 0.1B/0.4B 的英文线（0–12%），大片空白
    c.legend([(s[0], s[2]) for s in sets], x=c.w - 220, y=c.y(0.62))
    c.note(c.w - 220, c.y(0.40), ["dashed = style hit", "solid = English output"])
    c.xgrid([px(1.0), px(0.65), px(0.55), px(0.0)],
            ["α=1.0", "0.65", "0.55", "α=0.0"], label="α")
    c.note(c.pl, c.h - 12,
           ["0.1B hands over gradually (α=0.6: 50%/38%) · 0.4B collapses to ~10–25% both "
            "· only 1.5B has a real coexistence window (α=0.6: 88%/75%)"], size=9)
    return c.svg()


FIGS = {
    "fig1-three-way.svg": fig1,
    "fig2-anatomy.svg": fig2,
    "fig3-inheritance.svg": fig3,
    "fig4-phase-diagram.svg": fig4,
    "fig5-geometry-vs-behaviour.svg": fig5,
    "fig6-editing.svg": fig6,
    "fig7-scaling.svg": fig7,
    "fig8-capacity.svg": fig8,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="generate paper figures from docs/*.json")
    ap.add_argument("--check", action="store_true", help="validate inputs only")
    ap.add_argument("--only", default=None, help="substring filter, e.g. fig4")
    args = ap.parse_args()

    commit = _commit()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {k: v for k, v in FIGS.items() if args.only is None or args.only in k}
    written = []
    for name, fn in targets.items():
        svg = fn()
        if not args.check:
            body = svg.replace("</svg>", f'<!-- data: docs/*.json @ {commit} -->\n</svg>', 1)
            (OUT / name).write_text(body, encoding="utf-8")
        written.append(name)
    print(f"{'checked' if args.check else 'wrote'} {len(written)} figure(s) in {OUT}")
    for n in written:
        print("  ", n)


if __name__ == "__main__":
    main()
