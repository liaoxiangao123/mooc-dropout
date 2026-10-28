"""生成论文中的三张示意图（300 dpi，超采样抗锯齿）。

对应论文：
    图 3-1  观察窗与标签时点示意
    图 4-1  代价敏感 CatBoost 总体框架
    图 6-1  系统分层架构

Pillow 不做复杂文字整形，组合字符（p̂、ŷ）与 Unicode 下标（t₀）在微软雅黑
里缺字形，会渲染成方框。因此本脚本自带一个极简富文本绘制器：

    _{...}     下标
    ^{...}     上标
    \\hat{x}    给单个字符加帽子

用法：
    python scripts/make_schematic_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 基础参数

DPI = 300
SS = 2                               # 超采样倍数
FONT_DIR = Path("C:/Windows/Fonts")
SANS = "msyh.ttc"
SANS_B = "msyhbd.ttc"

# 学术风格配色：主蓝、强调橙，红/绿仅用于语义标注
INK = "#1F2A37"
NAVY = "#1F3864"
BLUE_FILL = "#E8EEF7"
BLUE_LINE = "#7E9BC2"
BLUE_TAB = "#5B7BA6"
ACCENT = "#C55A11"
ACCENT_FILL = "#FBEDE3"
ACCENT_TXT = "#9C4A1C"
GREY = "#6B7280"
GREY_FILL = "#F4F5F7"
GREY_LINE = "#C3C8D0"
GREEN_FILL = "#E7F1E9"
GREEN_LINE = "#6E9C74"
GREEN_TXT = "#33543A"
RED_FILL = "#FBE7E4"
RED_LINE = "#C0705F"
RED_TXT = "#9C4A3C"
WHITE = "#FFFFFF"

_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(name: str, pt: float) -> ImageFont.FreeTypeFont:
    px = max(1, int(round(pt * DPI / 72 * SS)))
    key = (name, px)
    if key not in _CACHE:
        _CACHE[key] = ImageFont.truetype(str(FONT_DIR / name), px)
    return _CACHE[key]


# ---------------------------------------------------------------- 富文本解析

SUB_SCALE = 0.70
SUP_SCALE = 0.70


def parse_rich(s: str) -> list[tuple[str, str]]:
    """把带标记的字符串拆成 (样式, 文本) 序列。

    样式取值：base / sub / sup / hat（帽子加在随后的 base 上）。
    """
    runs: list[tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("_{", i):
            j = s.index("}", i)
            runs.append(("sub", s[i + 2 : j]))
            i = j + 1
        elif s.startswith("^{", i):
            j = s.index("}", i)
            runs.append(("sup", s[i + 2 : j]))
            i = j + 1
        elif s.startswith("\\hat{", i):
            j = s.index("}", i)
            runs.append(("hat", s[i + 5 : j]))
            i = j + 1
        else:
            j = i
            while j < n and not (
                s.startswith("_{", j) or s.startswith("^{", j) or s.startswith("\\hat{", j)
            ):
                j += 1
            runs.append(("base", s[i:j]))
            i = j
    return [r for r in runs if r[1]]


def run_font(style: str, pt: float, bold: bool) -> ImageFont.FreeTypeFont:
    if style == "sub":
        return font(SANS_B if bold else SANS, pt * SUB_SCALE)
    if style == "sup":
        return font(SANS_B if bold else SANS, pt * SUP_SCALE)
    if style == "hat":
        return font(SANS_B if bold else SANS, pt)
    return font(SANS_B if bold else SANS, pt)


# ---------------------------------------------------------------- 画布


class Canvas:
    """以英寸为单位绘图，内部换算为超采样像素。"""

    def __init__(self, width_in: float, height_in: float) -> None:
        self.w_in = width_in
        self.h_in = height_in
        self.w = int(round(width_in * DPI * SS))
        self.h = int(round(height_in * DPI * SS))
        self.img = Image.new("RGB", (self.w, self.h), WHITE)
        self.d = ImageDraw.Draw(self.img)

    # -- 单位换算
    def x(self, inch: float) -> float:
        return inch * DPI * SS

    def y(self, inch: float) -> float:
        return inch * DPI * SS

    def lw(self, pt: float) -> int:
        return max(1, int(round(pt * DPI / 72 * SS)))

    def _px2in(self, px: float) -> float:
        return px / (DPI * SS)

    # -- 图元
    def box(self, x, y, w, h, *, fill=WHITE, outline=INK, width=1.0, radius=0.05, dash=False):
        px = [self.x(x), self.y(y), self.x(x + w), self.y(y + h)]
        if dash:
            self._dashed_round_rect(px, self.x(radius), outline, width)
        else:
            self.d.rounded_rectangle(
                px, radius=self.x(radius), fill=fill, outline=outline, width=self.lw(width)
            )
        return x, y, w, h

    def _dashed_round_rect(self, px, r, color, width) -> None:
        x0, y0, x1, y1 = px
        self._dashed((x0 + r, y0), (x1 - r, y0), color, width)
        self._dashed((x0 + r, y1), (x1 - r, y1), color, width)
        self._dashed((x0, y0 + r), (x0, y1 - r), color, width)
        self._dashed((x1, y0 + r), (x1, y1 - r), color, width)

    def _dashed(self, p0, p1, color, width, dash=0.075, gap=0.055) -> None:
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        d, g = self.x(dash), self.x(gap)
        pos = 0.0
        while pos < length:
            end = min(pos + d, length)
            self.d.line(
                [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
                fill=color,
                width=self.lw(width),
            )
            pos = end + g

    def line(self, x0, y0, x1, y1, *, color=INK, width=1.0, dash=False) -> None:
        p0, p1 = (self.x(x0), self.y(y0)), (self.x(x1), self.y(y1))
        if dash:
            self._dashed(p0, p1, color, width)
        else:
            self.d.line([p0, p1], fill=color, width=self.lw(width))

    def arrow(self, x0, y0, x1, y1, *, color=INK, width=1.1, head=0.075, dash=False) -> None:
        self.line(x0, y0, x1, y1, color=color, width=width, dash=dash)
        ang = math.atan2(y1 - y0, x1 - x0)
        hl = self.x(head)
        hw = hl * 0.45
        tip = (self.x(x1), self.y(y1))
        left = (
            tip[0] - hl * math.cos(ang) + hw * math.sin(ang),
            tip[1] - hl * math.sin(ang) - hw * math.cos(ang),
        )
        right = (
            tip[0] - hl * math.cos(ang) - hw * math.sin(ang),
            tip[1] - hl * math.sin(ang) + hw * math.cos(ang),
        )
        self.d.polygon([tip, left, right], fill=color)

    def text(self, x, y, s, *, pt=8.5, color=INK, bold=False, anchor="mm") -> None:
        self.d.text(
            (self.x(x), self.y(y)), s, font=font(SANS_B if bold else SANS, pt), fill=color, anchor=anchor
        )

    def rich(self, x, y, s, *, pt=8.5, color=INK, bold=False, anchor="mm") -> None:
        """带上下标/帽子的文本。y 为基线位置（anchor 含 s 时）。"""
        runs = parse_rich(s)
        # 量测总宽（像素）
        widths = []
        for style, txt in runs:
            f = run_font(style, pt, bold)
            bb = f.getbbox(txt)
            widths.append(max(bb[2] - bb[0], 0) + f.getbbox(" ")[2] * 0)
        total = sum(widths)

        # 水平起点：mm/lm 均按居中或左对齐处理
        start_px = self.x(x) - (total / 2 if anchor.startswith("m") else 0)
        baseline = self.y(y)

        cursor = start_px
        for (style, txt), wpx in zip(runs, widths):
            f = run_font(style, pt, bold)
            dy = 0.0
            if style == "sub":
                dy = self.y(0.048)
            elif style == "sup":
                dy = -self.y(0.040)
            self.d.text((cursor, baseline + dy), txt, font=f, fill=color, anchor="ls")
            if style == "hat":
                cx = cursor + wpx / 2
                top = baseline - self.y(0.088)
                self.d.line(
                    [(cx - wpx * 0.22, top + self.y(0.032)), (cx, top), (cx + wpx * 0.22, top + self.y(0.032))],
                    fill=color,
                    width=self.lw(pt * 0.09),
                    joint="curve",
                )
            cursor += wpx

    def rich_center_v(self, cx, cy, s, *, pt=8.5, color=INK, bold=False) -> None:
        """垂直方向也居中的富文本（按基线偏移修正）。"""
        self.rich(cx, cy + pt * 0.0012 * 8, s, pt=pt, color=color, bold=bold, anchor="mm")

    def lines(self, cx, cy, items, *, leading=0.19) -> None:
        """多行（每行可为富文本），整体以 (cx, cy) 为中心。"""
        n = len(items)
        start = cy - leading * (n - 1) / 2
        for i, (s, kw) in enumerate(items):
            kw = {"pt": 8.0, "color": INK, **kw}
            self.rich(cx, start + i * leading, s, anchor="mm", **kw)

    def save(self, path: Path) -> Path:
        out = self.img.resize(
            (int(self.w / SS), int(self.h / SS)), Image.LANCZOS
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path, dpi=(DPI, DPI), optimize=True)
        return path


# ---------------------------------------------------------------- 图 3-1


def figure_3_1(path: Path) -> Path:
    c = Canvas(6.1, 2.20)

    ax = 1.02            # 轴线纵坐标
    x0, x1, x2 = 0.92, 2.72, 4.86
    x_end = 5.58

    # 时间轴
    c.line(0.30, ax, x_end - 0.10, ax, color=INK, width=1.3)
    c.arrow(x_end - 0.12, ax, x_end, ax, color=INK, width=1.3, head=0.065)
    c.text(x_end + 0.02, ax - 0.17, "时间 t", pt=8, color=GREY, anchor="rm")

    # 观察窗（特征可用）
    c.box(x0, ax - 0.46, x1 - x0, 0.36, fill=BLUE_FILL, outline=BLUE_LINE, radius=0.035)
    c.lines(
        (x0 + x1) / 2,
        ax - 0.28,
        [("观察窗  W = 14 d", {"pt": 8.4, "bold": True, "color": NAVY}),
         ("特征 X 仅用此段事件计算", {"pt": 7.6, "color": "#3E5C86"})],
        leading=0.165,
    )

    # 禁止区（防穿越）
    c.box(x1, ax - 0.46, x2 - x1, 0.36, fill=RED_FILL, outline=RED_LINE, radius=0.035, dash=True)
    c.lines(
        (x1 + x2) / 2,
        ax - 0.28,
        [("不参与特征计算", {"pt": 8.4, "bold": True, "color": RED_TXT}),
         ("防止未来信息泄漏", {"pt": 7.6, "color": "#A2695C"})],
        leading=0.165,
    )

    # 时点刻度
    for xx, label in ((x0, "t_{0}"), (x1, "t_{0} + 14 d"), (x2, "课程结束")):
        c.line(xx, ax, xx, ax + 0.13, color=GREY, width=1.0)
        c.rich(xx, ax + 0.30, label, pt=8.4, bold=True, color=INK, anchor="mm")

    # 特征时点 / 标签时点括号
    for xa, xb, color, label in (
        (x0, x1, BLUE_LINE, "特征时点"),
        (x1, x2, ACCENT, "标签时点  y ∈ {0, 1}"),
    ):
        yb = ax + 0.46
        c.line(xa, yb, xb, yb, color=color, width=1.5)
        c.line(xa, yb - 0.06, xa, yb + 0.06, color=color, width=1.5)
        c.line(xb, yb - 0.06, xb, yb + 0.06, color=color, width=1.5)
        c.text((xa + xb) / 2, yb + 0.16, label, pt=7.8, bold=True, color=color)

    # 结论条
    c.box(0.30, 1.78, 5.50, 0.34, fill=GREY_FILL, outline=GREY_FILL, radius=0.035)
    c.text(3.05, 1.95, "特征时点严格早于标签时点：任一特征都不含观察窗之后的信息", pt=8.2, color=INK)
    return c.save(path)


# ---------------------------------------------------------------- 图 4-1


def figure_4_1(path: Path) -> Path:
    c = Canvas(6.1, 3.02)

    bw, bh, gap = 1.24, 0.62, 0.26
    left = 0.18
    row1, row2 = 0.26, 1.58
    cx = [left + i * (bw + gap) + bw / 2 for i in range(4)]

    top = [
        ("行为日志", "events.csv"),
        ("特征构建", "44 维特征"),
        ("CatBoost 训练", "正类权重 w = ρ"),
        ("概率输出", "\\hat{p} = P(y=1|x)"),
    ]
    bottom = [
        ("分级与解释", "高/中/低 + SHAP"),
        ("决策", "\\hat{y} = 1[\\hat{p} ≥ t^{*}]"),
        ("阈值 t^{*}", "OOF 上取最小代价"),
        ("期望代价最小", "C_{FN}·FN + C_{FP}·FP"),
    ]

    for x, (t, s) in zip(cx, top):
        c.box(x - bw / 2, row1, bw, bh, fill=BLUE_FILL, outline=BLUE_LINE, radius=0.055)
        c.rich(x, row1 + 0.27, t, pt=9.0, bold=True, color=NAVY)
        c.rich(x, row1 + 0.45, s, pt=7.2, color="#41597D")

    for x, (t, s) in zip(cx, bottom):
        c.box(x - bw / 2, row2, bw, bh, fill=ACCENT_FILL, outline=ACCENT, radius=0.055)
        c.rich(x, row2 + 0.27, t, pt=9.0, bold=True, color=ACCENT_TXT)
        c.rich(x, row2 + 0.45, s, pt=7.2, color="#A2694A")

    # 上排左→右
    for i in range(3):
        y = row1 + bh / 2
        c.arrow(cx[i] + bw / 2 + 0.02, y, cx[i + 1] - bw / 2 - 0.02, y, color=GREY, width=1.1, head=0.058)
    # 下排右→左
    for i in range(3):
        y = row2 + bh / 2
        c.arrow(cx[i + 1] - bw / 2 - 0.02, y, cx[i] + bw / 2 + 0.02, y, color=GREY, width=1.1, head=0.058)
    # 概率输出 ↓ 期望代价最小
    c.arrow(cx[3], row1 + bh + 0.02, cx[3], row2 - 0.02, color=GREY, width=1.1, head=0.058)
    c.lines(
        cx[3] - 0.58,
        (row1 + bh + row2) / 2,
        [("在候选阈值上", {"pt": 7.0, "color": GREY}),
         ("扫描期望代价", {"pt": 7.0, "color": GREY})],
        leading=0.155,
    )

    # 代价矩阵图例条
    c.box(0.30, 2.44, 5.50, 0.32, fill=WHITE, outline=ACCENT, radius=0.03, dash=True)
    c.rich(
        3.05,
        2.60,
        "代价矩阵标定：C_{FN} = 1000，C_{FP} = 100  →  同时决定训练权重 w 与决策阈值 t^{*}",
        pt=7.8,
        bold=True,
        color=ACCENT_TXT,
    )

    # 训练/决策两段标注
    c.text(2.72, 1.09, "① 训练：让模型偏向正类", pt=7.2, color=GREY, anchor="mm")
    c.text(2.72, 2.30, "② 决策：把阈值当作待优化参数", pt=7.2, color=GREY, anchor="mm")
    return c.save(path)


# ---------------------------------------------------------------- 图 6-1


def figure_6_1(path: Path) -> Path:
    c = Canvas(6.1, 3.62)

    bx, bw = 0.66, 4.62
    tab_w = 1.00
    row_h = 0.58
    gap = 0.16
    y0 = 0.30

    layers = [
        ("接入层", ["FastAPI 路由", "request_id 中间件", "全局异常处理"], BLUE_FILL, BLUE_LINE, BLUE_TAB, NAVY),
        ("服务层", ["PredictionService", "事件 → 特征 → 概率 → 分级 → 因子"], "#EDF2F9", BLUE_LINE, BLUE_TAB, NAVY),
        ("仓储层", ["ArtifactRepository", "模型 / 特征构建器 / 元数据缓存"], GREY_FILL, GREY_LINE, "#8C939F", "#374151"),
        ("领域层", ["FeatureBuilder · CatBoostRiskModel", "CostMatrix · RiskExplainer · Trainer"], GREEN_FILL, GREEN_LINE, "#5E8C66", GREEN_TXT),
    ]

    for i, (name, items, fill, outline, tabc, fg) in enumerate(layers):
        y = y0 + i * (row_h + gap)
        c.box(bx, y, bw, row_h, fill=fill, outline=outline, radius=0.035)
        # 左侧层名色块（右侧拼一个直角矩形，避免出现圆角缺口）
        c.box(bx, y, tab_w, row_h, fill=tabc, outline=tabc, radius=0.035)
        c.d.rectangle([c.x(bx + tab_w - 0.07), c.y(y), c.x(bx + tab_w), c.y(y + row_h)], fill=tabc)
        c.text(bx + tab_w / 2, y + row_h / 2, name, pt=9.6, bold=True, color=WHITE)
        # 内容按行排布
        n = len(items)
        lead = 0.175
        start = y + row_h / 2 - lead * (n - 1) / 2
        for j, s in enumerate(items):
            c.rich(bx + tab_w + 0.18, start + j * lead + 0.035, s, pt=8.0, color=fg, anchor="lm")

    # 层间箭头
    for i in range(3):
        yb = y0 + i * (row_h + gap) + row_h
        c.arrow(bx + bw / 2, yb + 0.015, bx + bw / 2, yb + gap - 0.02, color=GREY, width=1.0, head=0.05)

    # 左侧单向依赖箭头
    top_y = y0 + 0.05
    bot_y = y0 + 3 * (row_h + gap) + row_h - 0.05
    c.arrow(0.40, top_y, 0.40, bot_y, color=ACCENT, width=1.3, head=0.062)
    c.rich(0.40, top_y - 0.11, "依赖方向", pt=7.6, bold=True, color=ACCENT_TXT)

    # 底部说明
    c.box(0.30, 3.16, 5.50, 0.34, fill=GREY_FILL, outline=GREY_FILL, radius=0.035)
    c.text(
        3.05,
        3.33,
        "依赖单向：上层可调用下层，下层不反向依赖 —— 训练脚本因此可绕过 HTTP 复用同一套代码",
        pt=7.9,
        color=INK,
    )
    return c.save(path)


# ---------------------------------------------------------------- 入口


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "figs"
    for p in (
        figure_3_1(out / "fig3-1-observation-window.png"),
        figure_4_1(out / "fig4-1-framework.png"),
        figure_6_1(out / "fig6-1-architecture.png"),
    ):
        print(f"{p.name:36s} {p.stat().st_size:>8d} bytes")


if __name__ == "__main__":
    main()
