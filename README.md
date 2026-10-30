# MOOC 辍学风险预测（代价敏感 CatBoost）

> 毕业设计原型系统：基于 KDD Cup 2015（学堂在线）学习行为数据，预测选课学习者的辍学风险。
> 核心不是「把 AUC 刷高」，而是**把预测准确升级为决策代价最小**。

## 1. 这个项目解决什么问题

MOOC 的辍学率长期高企（KDD Cup 2015 数据中多数课程完成率不足 15%）。对教学管理者而言，真正的问题不是「谁可能辍学」，而是：

- 干预资源有限（教师工时、辅导名额），不可能给所有人都打电话；
- 漏掉一个真正会辍学的学生，损失远大于误报一次；
- 模型输出的概率必须能落到「预警 / 关注 / 不干预」三档决策上。

因此本项目把**非对称误分类代价**写进模型训练与决策阈值：漏报代价 `C_FN = 1000 元`、误报代价 `C_FP = 100 元`，用代价比驱动类别加权，并用期望代价最小化取代 0.5 阈值。

## 2. 快速开始（3 条命令）

```bash
# 1) 安装依赖（建议使用虚拟环境）
pip install -r requirements-dev.txt
#    需要逐位复现论文数值时，改用锁定版本安装：
#    pip install -r requirements.lock.txt

# 2) 无数据也能跑：生成同构合成数据并训练
python scripts/train.py --auto-synthetic

# 3) 启动演示（自动挑端口、等就绪、开浏览器），浏览器打开后即为演示页
python scripts/demo.py
```

> **演示用哪套模型？** 默认 `artifacts/model` 是合成数据主模型（阈值 0.1253），
> `artifacts/realdata_official` 是真实数据模型（阈值 0.1048，AUC 0.8407）。两套都能跑，取舍不同：
>
> | | 合成模型（默认） | 真实数据模型 |
> |---|---|---|
> | 单条预测 | 5.6% 低 / 50.0% 高，有对比 | 34.5% 高 / 89.4% 高，都过线 |
> | 12 人名单 | 6 高 + 6 中 | **12 人全高** |
> | 概率形态 | 保序回归台阶值（18~90 条事件锁在 0.0774） | 连续，随活跃度单调下降 |
> | 适合现场演示 | ✅ 有高低对比 | ❌ 一片红，像系统没用 |
>
> 真实模型全员过线不是 bug：辍学率 79%，阈值 0.1048 落在分布左尾外，正是 5.9 节讲的
> 「决策退化」。所以现场演示用合成模型，真实模型留到答问环节。
>
> ```bash
> python scripts/demo.py --artifacts-dir artifacts/realdata_official   # 切真实数据模型
> ```
>
> 动线、话术与七个追问的答法见 [docs/12-原型演示脚本.md](docs/12-原型演示脚本.md)。
> 用命令行参数而不是环境变量，是因为 Windows 下 cmd 与另一种 shell 设环境变量的写法不同，现场容易出错。

使用真实 KDD Cup 2015 数据：

```bash
# 数据已在 data/raw/kdd2015/（120,542 条选课 / 8,157,277 条事件 / 辍学率 79.29%）
python scripts/check_data.py --data-dir data/raw/kdd2015   # 先体检
python scripts/train.py --data-dir data/raw/kdd2015 --calibration isotonic
```

**真实数据上有两件事必须先想清楚**（详见 [docs/11](docs/11-数据获取与接入指南.md) 第七节）：

1. 官方辍学率 **79.29%**，辍学是**多数类**。ρ=10 时 Elkan 阈值只有 0.0909，
   于是代价最优决策退化为「给所有人报警」，人均代价恰好等于平凡基线 —— 这不是模型的错，
   而是代价参数与基率不匹配。所以必须同时报告 `cost_reduction_vs_trivial`。
2. 想换标签口径（早期辍学）也仍然是多数类（67.63%），高基率是 MOOC 数据的结构性特征。
   此时唯一还能给出可执行结论的是**容量视角**。

```bash
# 派生早期辍学标签（流式扫描 8.16M 条事件，约 25 s）
python scripts/derive_early_dropout.py --data-dir data/raw/kdd2015

# 两个分析脚本：读落盘的 predictions_*.csv，不需要重新训练
python scripts/cost_ratio_scan.py --auto      # 退化解在哪个 ρ 之后消失
python scripts/capacity_analysis.py --auto    # 限额干预 top-K：名单开多大、能抓多少
```

运行测试：

```bash
pytest -q          # 115 个用例
```

运行论文实验（基线对比 / 代价比敏感性 / 特征消融）：

```bash
python scripts/run_experiments.py --auto-synthetic --iterations 300
# 产出：reports/exp_comparison.csv、exp_cost_ratio.csv、exp_feature_ablation.csv
```

补充实验（多种子 / Bootstrap / 校准）与超参搜索：

```bash
python scripts/run_extra_experiments.py          # E6~E8
python scripts/tune.py --trials 30               # E9，目标 = OOF 期望代价
python scripts/train.py --data-dir data/synthetic --calibration isotonic
```

重新排版论文（Markdown → Word → PDF）：

```bash
python scripts/md_to_docx.py                     # 生成 docx：封面 + 独创性声明 + 摘要/目录 + 正文
powershell -ExecutionPolicy Bypass -File scripts/export_pdf.ps1   # Word 更新目录域后导出 PDF
```

第二步不能省：docx 里的目录是 TOC 域，不更新就导出，读者看到的是一句「右键更新域」的占位提示。封面信息（校名/学院/学号/指导教师）改 `docs/封面信息.json`，不必动排版代码。

超参搜索在真实数据上重跑（E9 / E9b）：产物按 `--tag` 分开存放，不会覆盖合成数据那份：

```bash
# 20 个 trial × 3 折 OOF，约 50 分钟；K 折特征只算一次，所有 trial 复用
python scripts/tune.py --data-dir data/raw/kdd2015 --tag realdata --trials 20 --timeout 5400
# 测试集上做「默认 vs 调参」的配对 Bootstrap（收益是否可分辨）
python scripts/validate_tuning.py --data-dir data/raw/kdd2015 --params reports/optuna/realdata/best_params.json
# 产出：reports/optuna/realdata/{trials.csv,best_params.json}、
#       reports/调参实验-E9-真实数据.md、reports/调参验证-E9b-真实数据.md
```

真实数据实验（KDD Cup 2015）：

```bash
# E13 五基线横向对比：两种标签口径各一遍（每模型 4 次拟合 = 3 折 OOF + 全量重训）
python scripts/run_experiments.py --data-dir data/raw/kdd2015 --skip-sensitivity --skip-ablation
python scripts/run_experiments.py --data-dir data/raw/kdd2015 --label-file data/processed/enrollment_early.csv \
    --skip-sensitivity --skip-ablation
python scripts/plot_baseline_realdata.py          # 图 5-8

# E14 多种子重复：5 种子 × 4 配置 × 2 口径 = 40 次训练，160 次拟合
# 注意：单进程跑，不要两个口径并行——每进程峰值内存约 3 GB，并行会 bad allocation
python scripts/multiseed_realdata.py --tag official --seeds 42 7 2024 1234 3407
python scripts/multiseed_realdata.py --tag early --label-file data/processed/enrollment_early.csv \
    --seeds 42 7 2024 1234 3407
python scripts/collect_multiseed.py               # 从已落盘 npz 重建 raw/summary/paired，不重训
python scripts/plot_multiseed_realdata.py         # 图 5-9
```

## 3. 项目结构

```
mooc-dropout/
├── src/mooc_dropout/
│   ├── config.py            # 环境变量集中配置 + 启动校验
│   ├── errors.py            # 类型化错误体系
│   ├── logging_config.py    # 结构化 JSON 日志（带 request_id）
│   ├── schemas.py           # HTTP 层数据契约（pydantic v2）
│   ├── data/                # 加载器、数据契约、分组切分
│   ├── features/            # 44 个行为/时序特征 + 特征字典
│   ├── models/              # CatBoost 核心模型 + 3 个 sklearn 基线
│   ├── evaluation/          # 代价矩阵、精确阈值搜索、指标、容量分析
│   ├── training/            # 训练编排、对比/消融实验
│   ├── explain/             # 原生 SHAP 解释（无 shap 依赖）
│   ├── repository/          # 模型产物仓储
│   ├── service/             # 业务服务层（不依赖 HTTP）
│   └── api/                 # FastAPI 控制器、中间件、演示页
├── scripts/                 # 生成数据 / 训练 / 实验 / 调参 / 容量与代价比分析 / 启动
├── tests/                   # 115 个自动化测试用例
├── requirements.lock.txt    # 实测通过的精确版本（复现用）
├── docs/                    # 13 份项目文档（00~12）+ 论文初稿（md/docx/pdf）与 13 张插图
├── artifacts/               # 模型产物（不提交：体积大，可用脚本重生成）
└── reports/                 # 指标与实验台账（csv/json 与手写报告已提交，
                             #   只忽略 *.png 与 coverage/ —— 论文里的数字要能对着仓库查）
```

## 4. 核心技术点

| 环节 | 做法 | 为什么重要 |
|---|---|---|
| 特征 | 44 维行为特征，含 14 天观察窗 + 指数时间衰减 | 防数据穿越，且刻画「学习持续性」 |
| 类别特征 | `course_id` / `username` 交给 CatBoost 原生处理 | 高基数类别免 One-Hot，避免维度爆炸 |
| 代价敏感 | 正类权重 = `C_FN / C_FP` | 与最优阈值同源，可数学证明 |
| 决策 | OOF 预测上搜索期望代价最小阈值 | 阈值不在测试集上调，避免乐观偏差 |
| 评估 | 期望代价 + PR-AUC + Recall + 代价曲线 | AUC 高 ≠ 省钱，避免答辩被追问 |
| 解释 | CatBoost 原生 SHAP，输出 top-5 风险因子 | 教师需要知道「为什么预警这个学生」 |
| 校准 | 保序回归 / Platt，在 OOF 上拟合，阈值随之重选 | 加权会破坏概率校准，不校准则概率不可按字面理解 |
| 派单 | `/v1/intervention-list` 导出按风险排序的 CSV | 教师可直接拿去打电话，而不是看一堆概率 |

## 5. 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活探针 |
| GET | `/ready` | 就绪探针（模型是否加载） |
| POST | `/v1/predict` | 单条选课记录风险预测 |
| POST | `/v1/predict/batch` | 批量预测（≤500 条），生成干预名单 |
| POST | `/v1/intervention-list` | 导出干预名单 CSV（按风险降序，可直接派单） |
| GET | `/v1/model/info` | 模型版本、阈值、代价矩阵、指标 |
| GET | `/v1/model/features` | 特征字典（分组 + 业务含义） |
| GET | `/docs` | OpenAPI 交互文档 |

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"enrollment_id":"E1","course_id":"C01","username":"u1",
       "events":[{"time":"2014-03-01T20:00:00","source":"video","event":"play_video","object":"v1"}]}'
```

响应示例：

```json
{
  "enrollment_id": "E1",
  "risk_probability": 0.5229,
  "raw_probability": 0.9526,
  "calibrated": true,
  "risk_level": "high",
  "decision_threshold": 0.08,
  "top_factors": [{"feature": "late_event_ratio", "value": 0.0, "contribution": 0.91, "direction": "risk_up", "source": "shap"}],
  "model_name": "catboost",
  "model_version": "20261023T071217Z"
}
```

`risk_probability` 是校准后的概率（可直接按字面理解），`raw_probability` 是模型原始输出，供追溯。

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| [docs/00-项目总览与交付清单.md](docs/00-项目总览与交付清单.md) | 一页纸看懂项目与本周交付物 |
| [docs/01-项目计划书.md](docs/01-项目计划书.md) | 目标、技术路线、里程碑、Agentic 协作机制 |
| [docs/02-需求规格说明书.md](docs/02-需求规格说明书.md) | 功能/非功能需求与验收标准 |
| [docs/03-系统设计说明书.md](docs/03-系统设计说明书.md) | 分层架构、模块职责、API 契约、ADR |
| [docs/04-数据说明与特征工程.md](docs/04-数据说明与特征工程.md) | 数据集、标签口径、防穿越、44 个特征 |
| [docs/05-算法设计-代价敏感CatBoost.md](docs/05-算法设计-代价敏感CatBoost.md) | 代价矩阵推导、权重等价性证明、阈值决策 |
| [docs/06-实验方案与评估协议.md](docs/06-实验方案与评估协议.md) | RQ、基线、指标、实验矩阵、结果模板 |
| [docs/07-任务分解WBS与周检清单.md](docs/07-任务分解WBS与周检清单.md) | 60 项细粒度任务、周五检查清单 |
| [docs/08-测试计划与用例.md](docs/08-测试计划与用例.md) | 测试策略、用例表、覆盖率 |
| [docs/09-风险登记册与答辩QA.md](docs/09-风险登记册与答辩QA.md) | 风险应对、24 个答辩追问 |
| [docs/10-论文大纲与图表清单.md](docs/10-论文大纲与图表清单.md) | 章节结构、图表清单、写作排期（含规划/实际字数对照） |
| [docs/11-数据获取与接入指南.md](docs/11-数据获取与接入指南.md) | 真实数据下载、放置、验收与踩坑 |
| [docs/12-原型演示脚本.md](docs/12-原型演示脚本.md) | 现场演示动线：点哪里、看到什么数、说什么话、七个追问 |

## 7. 数据与合规说明

- 使用公开学术数据集 KDD Cup 2015（XuetangX），不含真实身份信息，`username` 为脱敏 ID；
- 仓库不提交原始数据与模型产物（`.gitignore` 已配置）；
- 真实密钥只通过环境变量注入，仓库仅保留 `.env.example`。
