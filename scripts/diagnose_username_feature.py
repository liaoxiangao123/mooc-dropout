"""诊断：username 作为类别特征，对拟合与泛化有什么影响。

背景：`features/builder.py` 把 `username` 列入 `CATEGORICAL_FEATURES`，训练时作为
CatBoost 原生类别特征传入。切分本身按 username 分组，测试集用户全部未见，因此
不构成评估泄漏；但把高基数（真实数据约 8 万）的学习者 ID 当特征是否有意义，需要
用数据回答，而不是靠直觉。

做法：同一份特征矩阵、同一套 3 折分组切分、同一随机种子，跑「含 username」与
「不含 username」两版，比较训练集 AUC（拟合能力）与 OOF AUC / PR-AUC（泛化）。

结论（2026-10-30，合成数据 4000 条、正类率 21.2%、username 基数 1265）：
    含 username   训练集 0.9438 | OOF 0.7616 | PR 0.4515
    剔除 username 训练集 0.9545 | OOF 0.7667 | PR 0.4573
即该特征没有带来任何泛化收益。差值在噪声量级，故结论按「无贡献」表述，
不宣称剔除一定更优。已记入 docs/07 第 27 项。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mooc_dropout.data.split import oof_group_splits  # noqa: E402
from mooc_dropout.features.builder import FeatureBuilder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
enroll = pd.read_csv(ROOT / "data/synthetic/enrollment_train.csv")
logs = pd.read_csv(ROOT / "data/synthetic/log_train.csv")

X = FeatureBuilder().fit_transform(logs, enroll)
y = enroll.set_index("enrollment_id")["dropout"].reindex(X.index).to_numpy()
print(f"特征矩阵 {X.shape}，正类率 {y.mean():.3f}，username 基数 {X['username'].nunique()}")

splits = oof_group_splits(enroll, n_splits=3, seed=42)


def run(cols: list[str], cats: list[str], tag: str) -> None:
    oof = np.zeros(len(X))
    train_auc = []
    for tr, va in splits:
        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_seed=42,
            cat_features=cats,
        )
        model.fit(X.iloc[tr][cols], y[tr])
        oof[va] = model.predict_proba(X.iloc[va][cols])[:, 1]
        train_auc.append(roc_auc_score(y[tr], model.predict_proba(X.iloc[tr][cols])[:, 1]))
    print(
        f"{tag:<22} 训练集 AUC {np.mean(train_auc):.4f} | "
        f"OOF AUC {roc_auc_score(y, oof):.4f} | OOF PR-AUC {average_precision_score(y, oof):.4f}"
    )


base = [c for c in X.columns if c != "username"]
run(list(X.columns), ["course_id", "username"], "含 username（现状）")
run(base, ["course_id"], "不含 username")
