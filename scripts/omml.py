"""把论文里出现的 LaTeX 子集转成 OMML（Word 原生公式）。

只覆盖论文实际用到的语法，不追求通用性：
    \\frac{}{}  \\hat{}  \\bar{}  \\sqrt{}  \\sum_{}^{}  \\min_{}  \\arg\\min_{}
    \\mathbf{}  \\mathrm{}  \\mathbb{}  \\mathcal{}  \\text{}
    \\left( \\right)  \\big \\Big  \\begin{cases}...\\end{cases}  \\tag{}
    希腊字母与常见关系符、\\, \\quad \\qquad 等间距命令

输出的片段是 ``<m:oMath>`` 的内部内容；调用方负责包外层元素。

设计要点：
* Word 的行内公式默认把求和的上下限排在右侧，这里统一在 ``m:naryPr`` 里写
  ``m:limLoc="undOvr"``，强制上下限上下堆叠，视觉上等同于独立公式。
* 不使用 ``m:oMathPara``，因为独立公式要靠制表位实现"居中 + 右侧编号"，
  而 ``oMathPara`` 会独占整段、吞掉制表位。
"""

from __future__ import annotations

from xml.sax.saxutils import escape

M_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

# ---------------------------------------------------------------- 符号表

SYMBOLS: dict[str, str] = {
    # 关系符
    r"\ge": "≥", r"\le": "≤", r"\neq": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\sim": "∼", r"\propto": "∝", r"\ll": "≪", r"\gg": "≫",
    r"\Longleftrightarrow": "⟺", r"\Leftrightarrow": "⇔", r"\Rightarrow": "⇒",
    r"\rightarrow": "→", r"\leftarrow": "←", r"\to": "→", r"\mapsto": "↦",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\cup": "∪", r"\cap": "∩",
    # 运算符
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\ast": "∗", r"\circ": "∘", r"\partial": "∂", r"\nabla": "∇",
    r"\infty": "∞", r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\mid": "|", r"\vert": "|", r"\ldots": "…", r"\dots": "…", r"\cdots": "⋯",
    # 希腊字母
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ", r"\kappa": "κ", r"\lambda": "λ",
    r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ", r"\chi": "χ", r"\psi": "ψ",
    r"\omega": "ω", r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ",
    r"\Lambda": "Λ", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Omega": "Ω", r"\Pi": "Π",
    r"\ell": "ℓ", r"\hbar": "ℏ",
    # 其它
    r"\prime": "′", r"\degree": "°", r"\emptyset": "∅", r"\forall": "∀",
    r"\exists": "∃", r"\neg": "¬", r"\wedge": "∧", r"\vee": "∨",
}

SPACES: dict[str, str] = {
    r"\,": "\u2009",      # 细间距
    r"\:": "\u2005",
    r"\;": "\u2005",
    r"\!": "",
    r"\ ": "\u00a0",
    r"\quad": "\u2003",
    r"\qquad": "\u2003\u2003",
}

# 直立显示的运算符名
UPRIGHT_FUNCS = {"log", "ln", "exp", "min", "max", "arg", "sin", "cos", "tan", "lim", "sup", "inf"}

NARY_CHARS = {"sum": "∑", "prod": "∏", "int": "∫"}

_ACCENTS = {
    "hat": "\u0302",
    "widehat": "\u0302",
    "tilde": "\u0303",
    "dot": "\u0307",
    "vec": "\u20d7",
}


# ---------------------------------------------------------------- XML 片段


def mr(text: str, *, upright: bool = False, bold: bool = False) -> str:
    """一个数学 run。默认斜体（OMML 对拉丁字母的默认行为）。"""
    props = []
    if upright:
        props.append("<m:nor/>")
    if bold:
        props.append('<m:sty m:val="b"/>')
    rpr = f"<m:rPr>{''.join(props)}</m:rPr>" if props else ""
    return f'<m:r>{rpr}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>' 


def m_frac(num: str, den: str) -> str:
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def m_sub(base: str, sub: str) -> str:
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def m_sup(base: str, sup: str) -> str:
    return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"


def m_subsup(base: str, sub: str, sup: str) -> str:
    return (
        f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
    )


def m_nary(chr_: str, sub: str, sup: str, body: str) -> str:
    return (
        "<m:nary>"
        f'<m:naryPr><m:chr m:val="{chr_}"/><m:limLoc m:val="undOvr"/>'
        '<m:grow m:val="1"/><m:subHide m:val="0"/><m:supHide m:val="0"/></m:naryPr>'
        f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e>"
        "</m:nary>"
    )


def m_delim(beg: str, end: str, body: str) -> str:
    return (
        "<m:d>"
        f'<m:dPr><m:begChr m:val="{escape(beg)}"/><m:endChr m:val="{escape(end)}"/></m:dPr>'
        f"<m:e>{body}</m:e></m:d>"
    )


def m_acc(chr_: str, body: str) -> str:
    return f'<m:acc><m:accPr><m:chr m:val="{chr_}"/></m:accPr><m:e>{body}</m:e></m:acc>'


def m_bar(body: str) -> str:
    return f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{body}</m:e></m:bar>'


def m_rad(body: str) -> str:
    return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{body}</m:e></m:rad>'


def m_limlow(base: str, lim: str) -> str:
    return f"<m:limLow><m:e>{base}</m:e><m:lim>{lim}</m:lim></m:limLow>"


def m_eqarr(rows: list[str]) -> str:
    body = "".join(f"<m:e>{r}</m:e>" for r in rows)
    return f"<m:eqArr>{body}</m:eqArr>"


# ---------------------------------------------------------------- 解析器


class LatexError(ValueError):
    """公式解析失败，调用方应降级为纯文本。"""


class OmmlParser:
    """递归下降解析器，产出 OMML 片段。"""

    def __init__(self, src: str) -> None:
        self.s = src
        self.i = 0
        self.n = len(src)

    # -- 游标
    def _skip_spaces(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t\n":
            self.i += 1

    def _at(self, token: str) -> bool:
        return self.s.startswith(token, self.i)

    def _eat(self, token: str) -> bool:
        if self._at(token):
            self.i += len(token)
            return True
        return False

    # -- 主入口
    def parse(self) -> str:
        out = self._seq()
        if self.i < self.n:
            raise LatexError(f"未消费的输入：{self.s[self.i:self.i + 20]!r}")
        return out

    def _seq(self, stops: frozenset[str] = frozenset()) -> str:
        """解析到指定停止符或作用域结尾。"""
        parts: list[str] = []
        while self.i < self.n:
            if any(self._at(t) for t in stops):
                break
            if self.s[self.i] in "}&":
                break
            parts.append(self._scripted(stops))
        return "".join(parts)

    def _scripted(self, stops: frozenset[str]) -> str:
        """原子 + 可选的上/下标。"""
        self._skip_spaces()
        if self.i >= self.n:
            return ""
        base = self._atom(stops)

        sub = sup = None
        while True:
            self._skip_spaces()
            if self._eat("_"):
                sub = self._group()
            elif self._eat("^"):
                sup = self._group()
            else:
                break

        if sub is not None and sup is not None:
            return m_subsup(base, sub, sup)
        if sub is not None:
            return m_sub(base, sub)
        if sup is not None:
            return m_sup(base, sup)
        return base

    def _atom(self, stops: frozenset[str]) -> str:
        self._skip_spaces()
        if self.i >= self.n:
            return ""
        c = self.s[self.i]

        if c == "{":
            self.i += 1
            inner = self._seq()
            if not self._eat("}"):
                raise LatexError("缺少右花括号")
            return inner
        if c == "\\":
            return self._command(stops)
        self.i += 1
        return mr(c)

    def _group(self) -> str:
        self._skip_spaces()
        if self._eat("{"):
            inner = self._seq()
            if not self._eat("}"):
                raise LatexError("缺少右花括号")
            return inner
        if self.i < self.n and self.s[self.i] == "\\":
            return self._command(frozenset())
        if self.i < self.n:
            ch = self.s[self.i]
            self.i += 1
            return mr("\u2217" if ch == "*" else ch)
        return ""

    # -- 命令
    def _command(self, stops: frozenset[str]) -> str:
        start = self.i
        self.i += 1  # 跳过反斜杠
        if self.i >= self.n:
            raise LatexError("孤立的反斜杠")

        # 符号类命令：\\ 加字母串
        j = self.i
        while j < self.n and (self.s[j].isalpha() or self.s[j] == "*"):
            j += 1
        if j == self.i:                       # \\, \\; \\! 等
            token = self.s[start : start + 2]
            self.i = start + 2
            if token in SPACES:
                return mr(SPACES[token]) if SPACES[token] else ""
            if token == r"\\":
                raise LatexError("换行命令出现在非法位置")
            if token in (r"\{", r"\}", r"\%", r"\$", r"\&", r"\_", r"\#"):
                return mr(token[1])
            raise LatexError(f"未知命令 {token}")

        name = self.s[self.i : j]
        self.i = j
        token = "\\" + name

        if token in SYMBOLS:
            return mr(SYMBOLS[token])
        if token in SPACES:
            val = SPACES[token]
            return mr(val) if val else ""

        # 结构类
        if name == "frac" or name == "dfrac" or name == "tfrac":
            return m_frac(self._group(), self._group())
        if name == "sqrt":
            return m_rad(self._group())
        if name in _ACCENTS:
            return m_acc(_ACCENTS[name], self._group())
        if name == "bar":
            return m_bar(self._group())
        if name in ("text", "textrm", "mbox"):
            return mr(self._raw_group(), upright=True)
        if name in ("mathrm", "operatorname"):
            return mr(self._raw_group(), upright=True)
        if name == "mathbf":
            return mr(self._raw_group(), bold=True)
        if name in ("mathbb", "mathcal", "mathscr", "mathfrak"):
            # Word 无对应字体分支，退化为直立/斜体，语义仍可读
            inner = self._raw_group()
            return mr(inner, upright=(name == "mathbb"))
        if name in ("left", "right"):
            return self._delimited()
        if name in ("big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr"):
            return ""                          # 尺寸修饰，交给 m:d 处理
        if name == "begin":
            return self._environment()
        if name == "tag" or name == "label" or name == "nonumber":
            self._raw_group()
            return ""
        if name == "sum" or name == "prod" or name == "int":
            return self._nary(NARY_CHARS[name], stops)
        if name in UPRIGHT_FUNCS:
            if name == "arg":
                # arg min / arg max
                self._skip_spaces()
                if self._at("\\min") or self._at("\\max"):
                    which = self.s[self.i + 1 : self.i + 4]
                    self.i += 4
                    return self._func_with_limits("arg " + which)
                return mr("arg", upright=True)
            return self._func_with_limits(name)
        raise LatexError(f"未知命令 {token}")

    def _raw_group(self) -> str:
        """取 {...} 的原始文本（用于 \\text 等需要保留字面量的场合）。"""
        self._skip_spaces()
        if not self._eat("{"):
            if self.i < self.n:
                ch = self.s[self.i]
                self.i += 1
                return ch
            return ""
        depth = 1
        buf: list[str] = []
        while self.i < self.n and depth > 0:
            ch = self.s[self.i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    break
            buf.append(ch)
            self.i += 1
        return "".join(buf)

    def _delimited(self) -> str:
        """\\left( ... \\right) ：自动伸缩的定界符。"""
        self._skip_spaces()
        beg = self._read_delim()
        inner = self._seq(frozenset({r"\right"}))
        if not self._eat(r"\right"):
            raise LatexError("\\left 缺少配对的 \\right")
        self._skip_spaces()
        end = self._read_delim()
        return m_delim(beg, end, inner)

    def _read_delim(self) -> str:
        self._skip_spaces()
        if self.i >= self.n:
            raise LatexError("缺少定界符")
        c = self.s[self.i]
        if c == "\\":
            # \{ \} \| \langle ...
            if self._at(r"\{"):
                self.i += 2
                return "{"
            if self._at(r"\}"):
                self.i += 2
                return "}"
            if self._at(r"\|"):
                self.i += 2
                return "‖"
            if self._at(r"\langle"):
                self.i += 7
                return "⟨"
            if self._at(r"\rangle"):
                self.i += 7
                return "⟩"
            if self._at(r"\lvert"):
                self.i += 6
                return "|"
            if self._at(r"\rvert"):
                self.i += 6
                return "|"
            raise LatexError("无法识别的定界符")
        if c == ".":
            self.i += 1
            return ""
        self.i += 1
        return c

    def _nary(self, chr_: str, stops: frozenset[str]) -> str:
        """求和/连乘：先读上下限，再把本作用域剩余内容整体当作被作用体。"""
        sub = sup = ""
        while True:
            self._skip_spaces()
            if self._eat("_"):
                sub = self._group()
            elif self._eat("^"):
                sup = self._group()
            else:
                break
        body = self._seq(stops)
        return m_nary(chr_, sub, sup, body)

    def _func_with_limits(self, name: str) -> str:
        """min/max/log 等：若紧跟下标则渲染为下限（如 argmin 的下标）。"""
        self._skip_spaces()
        if self._at("_"):
            self.i += 1
            return m_limlow(mr(name, upright=True), self._group())
        return mr(name, upright=True)

    def _environment(self) -> str:
        self._skip_spaces()
        env = self._raw_group()
        if env not in ("cases", "aligned", "array", "split"):
            raise LatexError(f"不支持的公式环境 {env}")

        # 逐行解析；行内以 & 分列，行间以 \\ 分隔
        rows: list[list[str]] = []
        while True:
            cells: list[str] = []
            while True:
                cells.append(self._seq(frozenset({r"\\", r"\end", "&"})))
                if self._eat("&"):
                    continue
                break
            rows.append(cells)
            if self._eat(r"\\"):
                continue
            if self._eat(r"\end"):
                self._raw_group()
                break
            raise LatexError(f"{env} 环境未闭合")

        row_xml: list[str] = []
        for cells in rows:
            parts: list[str] = []
            for k, cell in enumerate(cells):
                if k:
                    parts.append(mr("\u2003\u2003"))
                parts.append(cell)
            row_xml.append("".join(parts))

        if env == "cases":
            return m_delim("{", "", m_eqarr(row_xml))
        return m_eqarr(row_xml)


# ---------------------------------------------------------------- 对外接口


def latex_to_omml_inner(src: str) -> str:
    """返回 ``<m:oMath>`` 的内部内容。解析失败抛 LatexError。"""
    cleaned = src.strip()
    # 去掉 \tag{...}，编号由外层排版处理
    parser = OmmlParser(cleaned)
    return parser.parse()


def omml_element(src: str, *, display: bool = False) -> str:
    """返回可交给 parse_xml 的 ``<m:oMath>`` 字符串。"""
    inner = latex_to_omml_inner(src)
    return f"<m:oMath {M_NS} {W_NS}>{inner}</m:oMath>"
