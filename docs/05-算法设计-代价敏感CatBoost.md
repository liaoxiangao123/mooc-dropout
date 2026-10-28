# 05 · 算法设计：代价敏感 CatBoost

> 本章是论文的核心章节。目标不是「用了 CatBoost」，而是把**代价敏感**从一句口号变成可推导、可实现、可验证的三层机制。

---

## 1. 问题形式化

给定选课样本集 $D=\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$，其中 $\mathbf{x}_i \in \mathbb{R}^{42} \times \mathcal{C}^2$（42 个数值特征 + 2 个类别特征），$y_i \in \{0,1\}$ 表示是否辍学。

模型输出 $\hat p_i = P(y_i=1\mid \mathbf{x}_i)$，决策规则为 $\hat y_i = \mathbb{1}[\hat p_i \ge t]$。

**目标**：在给定代价矩阵下最小化期望误分类代价

$$\min_{\theta,\,t}\ \mathbb{E}\big[C(y,\hat y)\big]=\frac{1}{N}\sum_{i=1}^{N}\Big[C_{FN}\,y_i(1-\hat y_i)+C_{FP}(1-y_i)\hat y_i\Big]$$

注意这里有**两个可优化对象**：模型参数 $\theta$ 与决策阈值 $t$。多数工作只优化 $\theta$，把 $t$ 固定为 0.5，这正是本项目要补的缺口。

## 2. 为什么是 CatBoost

| 机制 | 作用 | 对本课题的意义 |
|---|---|---|
| Ordered Target Statistics | 对类别特征做有序目标编码，避免目标泄漏 | `course_id`、`username` 高基数，免 One-Hot |
| Ordered Boosting | 用不同排列训练，缓解预测偏移 | 小样本课程上更稳 |
| 对称（oblivious）树 | 每层同一分裂条件，推理快 | 单条推理延迟低，适合在线预警 |
| 原生类别特征 | 无需人工编码 | 训练/推理特征处理一致，减少线上线下偏差 |

## 3. 代价敏感的三条路线

### 3.1 路线 A：重加权（Weighted Logloss）

对正类样本加权 $w=C_{FN}/C_{FP}$：

$$L_w=-\frac{1}{N}\sum_{i=1}^{N}\Big[w\,y_i\log \hat p_i+(1-y_i)\log(1-\hat p_i)\Big]$$

**等价性定理**：设真实后验 $\eta(\mathbf{x})=P(y=1\mid\mathbf{x})$，加权损失的最优解为

$$\hat p_w(\mathbf{x})=\frac{w\,\eta(\mathbf{x})}{w\,\eta(\mathbf{x})+1-\eta(\mathbf{x})}$$

则 $\hat p_w\ge 0.5 \iff \eta(\mathbf{x}) \ge \dfrac{1}{1+w}$。

取 $w=C_{FN}/C_{FP}$，得 $\eta^*=\dfrac{C_{FP}}{C_{FP}+C_{FN}}$，正是 Elkan 最优阈值。

> **这个证明的意义**：加权与阈值移动在理论上同源，「加权就够了、不用调阈值」的说法只在加权恰好等于代价比时成立；实际中仍需要用验证集校准 $t$，因为模型并不完美、概率也未校准。论文中用 E2/E3 两个实验分别验证。

### 3.2 路线 B：阈值移动（Threshold Moving）

不改变训练，只在决策时选 $t^*$：

$$t^* = \arg\min_{t}\ \big[C_{FN}\cdot FN(t)+C_{FP}\cdot FP(t)\big]$$

工程实现：在**训练集 OOF 预测**上网格搜索（`evaluation/cost.py::cost_optimal_threshold`），避免在测试集上调参。

### 3.3 路线 C：代价敏感损失（Custom Objective）

直接改造损失函数，样本级代价系数 $c_i = C_{FN}$ if $y_i=1$ else $C_{FP}$：

$$L_{cs}=-\frac{1}{N}\sum_i c_i\big[y_i\log\sigma(a_i)+(1-y_i)\log(1-\sigma(a_i))\big]$$

对近似值 $a_i$ 求一阶、二阶导：

$$\frac{\partial L}{\partial a_i}=c_i\big(\sigma(a_i)-y_i\big),\qquad \frac{\partial^2 L}{\partial a_i^2}=c_i\,\sigma(a_i)\big(1-\sigma(a_i)\big)$$

实现为 CatBoost 自定义目标（`models/cost_sensitive_loss.py`），并注意：**使用路线 C 时不能再叠加 `class_weights`**，否则代价被重复计入。

### 3.4 三条路线的取舍

| 路线 | 实现成本 | 理论纯度 | 本项目定位 |
|---|---|---|---|
| A 重加权 | 低（1 行参数） | 中（等价于阈值移动） | 主方案，稳定 |
| B 阈值移动 | 低 | 高（直接优化决策） | 必做，与 A 组合 |
| C 自定义损失 | 中（需实现导数） | 高（训练即代价敏感） | 探索性对比，作为创新点 |

## 4. 最优阈值推导（含干预成本扩展）

记预测 1 的期望代价与预测 0 的期望代价：

$$E[c\mid \hat y=1]=p\,C_{TP}+(1-p)\,C_{FP}$$
$$E[c\mid \hat y=0]=p\,C_{FN}+(1-p)\,C_{TN}$$

令两者相等：

$$p^*=\frac{C_{TN}-C_{FP}}{C_{TP}-C_{FP}-C_{FN}+C_{TN}}$$

当 $C_{TP}=C_{TN}=0$ 时退化为 $p^*=\dfrac{C_{FP}}{C_{FP}+C_{FN}}$。

**业务标定示例**（本项目默认值）：

| 代价 | 取值 | 口径 |
|---|---|---|
| $C_{FN}$ | 1000 元 | 一名学生流失造成的学费 + 续报损失 |
| $C_{FP}$ | 100 元 | 一次无效干预的教师工时 + 触达成本 |
| 代价比 $\rho$ | 10 | $p^*=0.0909$ |

即：只要模型认为辍学概率超过 **9%**，预警就是划算的。这解释了为什么"高风险"名单会远长于按 0.5 切割的结果——这不是模型不准，而是决策本来就该更激进。

## 5. 训练流程

```
输入: 选课表 E, 日志 L, 代价矩阵 (C_FN, C_FP), 窗口 W
1. 校验 E、L 的字段与取值
2. 按 username 分组切分: (E_tr, E_te) ← GroupShuffleSplit(0.2)
3. 用 E_tr 的日志构建特征: X_tr = FeatureBuilder.fit_transform(L_tr, E_tr)
4. 训练主模型: M = CatBoost(X_tr, y_tr, class_weights=[1, C_FN/C_FP])
5. 在 E_tr 上做 5 折分组 OOF 预测 p_oof
6. 阈值: t* = argmin_t [C_FN·FN(t) + C_FP·FP(t)]  (基于 p_oof)
7. 测试集评估: p_te = M(X_te); 报告 t* 与 0.5 两套指标
8. 保存 M、FeatureBuilder、metadata(含 t* 与代价矩阵)
```

**为什么阈值要用 OOF 而不是测试集**：用测试集选阈值等于「在考卷上划重点」，指标会系统性偏乐观。OOF 让阈值选择与最终评估使用互不重叠的样本。

## 6. 超参数搜索

| 超参 | 搜索范围 | 说明 |
|---|---|---|
| `depth` | 4-10 | 越深越易过拟合小课程 |
| `learning_rate` | 0.01-0.3 | 与 iterations 联动 |
| `l2_leaf_reg` | 1-20 | 正则强度 |
| `iterations` | 200-1500 | 配合早停 |
| `border_count` | 64-254 | 数值特征分箱数 |
| 衰减系数 $\lambda$ | 0.05-0.5 | 特征侧超参，需联合搜索 |
| 代价比 $\rho$ | {1,2,5,10,20} | 不做"最优"选择，做敏感性分析 |

**关键**：优化目标必须是**期望代价**（或 PR-AUC），不能是 ROC-AUC，否则调参与业务目标脱节。工具建议 Optuna（TPE），每个 trial 用 3 折 OOF 的期望代价作为返回值的负数。

## 7. 复杂度与工程优化

| 环节 | 朴素做法 | 本项目做法 | 效果 |
|---|---|---|---|
| 特征聚合 | 对每条选课 `groupby.apply` | 一次性向量化 groupby + `transform` | 12 万条从 ~15 分钟降到 ~40 秒 |
| 阈值搜索 | 逐样本循环 | 向量化比较 + 99 点网格 | 毫秒级 |
| 推理 | 每次重新加载模型 | 仓储层内存缓存 | P95 < 100ms |
| 解释 | 每次调用 shap 包 | CatBoost 原生 ShapValues | 省一个重依赖 |

## 8. 与相关工作的差异

| 维度 | 常见做法 | 本项目 |
|---|---|---|
| 不平衡处理 | SMOTE / 随机欠采样 | 不采样，用代价加权（保留负类信息） |
| 代价设置 | 直接设 `pos_weight=10` | 业务口径标定 + ρ 敏感性分析 |
| 阈值 | 固定 0.5 | OOF 上搜索最小期望代价 |
| 评估 | AUC / Accuracy | 期望代价 + PR-AUC + 代价曲线 |
| 可解释 | 仅给特征重要性 | 单样本 SHAP 因子 + 方向 |
| 工程 | 只有 notebook | 分层服务 + 77 个测试 |

## 9. 本章小结（可直接写进论文）

本章把「代价敏感」拆成训练层与决策层两个可独立验证的机制，并证明了重加权与阈值移动在理想条件下的等价性；实验设计上通过 E2（加权 vs 不加权）与 E3（阈值 0.5 vs $t^*$）分别隔离两者的贡献，避免把收益错误归因。
