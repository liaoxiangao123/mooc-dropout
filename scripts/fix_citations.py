"""把正文的「作者-年份」引用改为顺序编码制 [n]（GB/T 7714）。

做法分三步，保证可审计：
    1. 正文中的引用先用不可混淆的标记 ``{{c:KEY,KEY}}`` 替换；
    2. 按标记在正文中首次出现的顺序分配编号；
    3. 按新编号重排参考文献表，并把标记渲染成 [1]、[3-5] 这类形式。

顺带修掉一个隐患：原稿 20 条文献里只有 9 条被正文引用过，其余属于「挂着不用」，
答辩时容易被追问。本脚本为每条文献在语义合适的位置补了引用，并断言最终
「每一条文献都被引用、每一个编号都有对应文献」。

用法：
    python scripts/fix_citations.py --dry-run   # 只看统计，不写文件
    python scripts/fix_citations.py             # 直接改写论文 md
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TITLE = "论文初稿-基于代价敏感CatBoost的MOOC辍学风险预测.md"

# 原文献编号 -> 引用键（键名取「作者+年份」，与本脚本的可读性一致）
KEY_OF_OLD_NUMBER = {
    1: "Elkan2001", 2: "CatBoost2018", 3: "XGB2016", 4: "LightGBM2017",
    5: "Friedman2001", 6: "SHAP2017", 7: "Drummond2006", 8: "HeGarcia2009",
    9: "Japkowicz2002", 10: "Saito2015", 11: "Brier1950", 12: "Zadrozny2001",
    13: "Niculescu2005", 14: "Kizilcec2013", 15: "Reich2019", 16: "Jordan2014",
    17: "Zhou2016", 18: "Li2019", 19: "KDD2015", 20: "Sklearn2011",
}

# (旧文本, 新文本) —— 前者必须精确匹配一次
REPLACEMENTS: list[tuple[str, str]] = [
    # 1.1.1 完成率
    ("（Reich & Ruipérez-Valiente, 2019；Jordan, 2014）",
     "{{c:Reich2019,Jordan2014}}"),
    # 1.2.1 行为模式
    ("Kizilcec 等（2013）", "Kizilcec 等{{c:Kizilcec2013}}"),
    # 1.2.2 三个代表实现
    ("Chen 与 Guestrin（2016）提出的 XGBoost、Ke 等（2017）提出的 LightGBM 以及 Prokhorenkova 等（2018）提出的 CatBoost",
     "Chen 与 Guestrin{{c:XGB2016}}提出的 XGBoost、Ke 等{{c:LightGBM2017}}提出的 LightGBM "
     "以及 Prokhorenkova 等{{c:CatBoost2018}}提出的 CatBoost"),
    # 1.2.3 不平衡与代价敏感
    ("类别不平衡是辍学预测的固有属性。",
     "类别不平衡是辍学预测的固有属性，其影响已有系统性实证{{c:Japkowicz2002}}。"),
    ("He 与 Garcia（2009）", "He 与 Garcia{{c:HeGarcia2009}}"),
    ("代价敏感学习由 Elkan（2001）系统化", "代价敏感学习由 Elkan{{c:Elkan2001}}系统化"),
    ("Drummond 与 Holte（2006）", "Drummond 与 Holte{{c:Drummond2006}}"),
    # 2.4.1 梯度提升 + 教科书
    ("梯度提升（Friedman, 2001）以加法模型逐步拟合负梯度：",
     "梯度提升{{c:Friedman2001}}以加法模型逐步拟合负梯度（该框架在{{c:Zhou2016}}与{{c:Li2019}}中有系统推导）："),
    # 2.5.1 PR-AUC
    ("Saito 与 Rehmsmeier（2015）", "Saito 与 Rehmsmeier{{c:Saito2015}}"),
    # 2.5.3 Brier
    ("**Brier 分数**定义为", "**Brier 分数**{{c:Brier1950}}定义为"),
    # 3.1.1 数据集出处
    ("是该领域最具代表性的公开基准之一。",
     "是该领域最具代表性的公开基准之一{{c:KDD2015}}。"),
    # 5.1.2 基线实现
    ("为验证「收益究竟来自模型结构还是代价机制」，设置五个模型，使用同一份特征矩阵与同一切分：",
     "为验证「收益究竟来自模型结构还是代价机制」，设置五个模型"
     "（其中三个 sklearn 基线基于 scikit-learn{{c:Sklearn2011}} 实现），使用同一份特征矩阵与同一切分："),
    # 表 5-8 个案解释
    ("模型输出风险概率 0.9526，判定为高风险，主要致险因子如表 5-8。",
     "模型输出风险概率 0.9526，判定为高风险；按 Shapley 值分解{{c:SHAP2017}}，主要致险因子如表 5-8。"),
    # 5.7.2 校准方法
    ("学术上的应对是引入 Platt 校准或保序回归，这已列入后续工作。",
     "学术上的应对是引入 Platt 校准{{c:Zadrozny2001}}或保序回归{{c:Niculescu2005}}，这已列入后续工作。"),
]


def compress(numbers: list[int]) -> str:
    """[1,2,3,7] -> "1-3,7" """
    nums = sorted(set(numbers))
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if j == i else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = DOCS / TITLE
    text = path.read_text(encoding="utf-8")
    body, refs_block = text.split("## 参考文献", 1)

    # ---- 解析原有文献表
    entries = {
        KEY_OF_OLD_NUMBER[int(n)]: desc.strip()
        for n, desc in re.findall(r"^\[(\d+)\]\s+(.+)$", refs_block, re.M)
    }
    if len(entries) != 20:
        print(f"!! 文献解析异常：得到 {len(entries)} 条，期望 20 条")
        return 1

    # ---- 第一步：替换为标记
    for old, new in REPLACEMENTS:
        count = body.count(old)
        if count != 1:
            print(f"!! 锚点匹配 {count} 次（应为 1 次）：{old[:42]}…")
            return 1
        body = body.replace(old, new)

    # ---- 第二步：按首次出现顺序编号
    order: list[str] = []
    for m in re.finditer(r"\{\{c:([A-Za-z0-9,]+)\}\}", body):
        for key in m.group(1).split(","):
            if key not in order:
                order.append(key)

    unknown = [k for k in order if k not in entries]
    if unknown:
        print(f"!! 引用了不存在的文献键：{unknown}")
        return 1
    uncited = [k for k in entries if k not in order]
    if uncited:
        print(f"!! 以下文献未被正文引用：{uncited}")
        return 1

    number = {key: i + 1 for i, key in enumerate(order)}

    # ---- 第三步：渲染标记
    def render(m: re.Match) -> str:
        keys = m.group(1).split(",")
        return "[" + compress([number[k] for k in keys]) + "]"

    new_body = re.sub(r"\{\{c:([A-Za-z0-9,]+)\}\}", render, body)

    # ---- 重建文献表
    lines = ["", ""]
    for key in order:
        lines.append(f"[{number[key]}] {entries[key]}")
        lines.append("")
    new_refs = "\n".join(lines)

    # 保留参考文献原区块尾部（附录标题前的内容）
    tail = refs_block.split("\n## ", 1)
    tail_part = ("\n## " + tail[1]) if len(tail) > 1 else ""

    out = new_body.rstrip() + "\n\n## 参考文献" + new_refs + "\n" + tail_part.lstrip("\n")

    # ---- 校验（在标记层面做，避免把正文里的数学区间 [0, 1] 误判为引用）
    assert set(number.values()) == set(range(1, len(order) + 1)), "编号出现空洞"
    assert len(number) == len(entries), "存在未编号的文献"
    remaining = re.findall(r"\{\{c:[^}]*\}\}", new_body)
    assert not remaining, f"仍有未渲染的标记：{remaining[:3]}"

    marker_count = len(re.findall(r"\{\{c:[A-Za-z0-9,]+\}\}", body))
    print(f"文献 {len(entries)} 条，全部被引用；正文共 {marker_count} 处引用标记")
    print("\n编号对照（首次出现顺序）：")
    for key in order:
        print(f"  [{number[key]:>2}] {key:<16} {entries[key][:52]}")

    if args.dry_run:
        print("\n--dry-run：未写入文件")
        return 0

    backup = path.with_suffix(".md.bak")
    shutil.copy(path, backup)
    path.write_text(out, encoding="utf-8")
    print(f"\n已写入 {path.name}（备份 {backup.name}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
