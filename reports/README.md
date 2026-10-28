# reports/ 索引

这个目录有 29 个条目、10 份实验报告，其中**只有 4 份是手写**，其余 6 份由脚本重写。
读之前先看这张表，避免两种情况：把过期的自动产物当结论引用；或者手改了某个文件，下次跑脚本又被覆盖。

## 一、实验报告（按论文顺序）

| 论文节 | 实验 | 报告 | 生成方式 | 数据源 |
|---|---|---|---|---|
| 5.2~5.6 | E1、E2、E4、E5 | `实验记录-合成数据.md` | **手写**（含分析与局限） | `exp_comparison.csv`、`exp_cost_ratio.csv`、`exp_feature_ablation.csv` |
| 5.7.3、5.8 | E6、E6b、E7、E7b、E8 | `补充实验-多种子与校准.md` | 自动（`scripts/run_extra_experiments.py`） | 同目录 `exp_multiseed*.csv`、`exp_bootstrap.csv`、`exp_paired_bootstrap.csv`、`exp_calibration.csv` |
| 5.9.2 | 早期辍学标签派生 | `早期辍学标签-统计.md` | 自动（`scripts/derive_early_dropout.py`） | `data/processed/enrollment_early.csv` |
| 5.9.4 | E11 代价比扫描 | `代价比扫描.md` | 自动（`scripts/cost_ratio_scan.py`） | `cost_ratio_scan.csv` |
| 5.9.5 | E12 容量分析 | `容量分析-K曲线.md` | 自动（`scripts/capacity_analysis.py`） | `capacity_curve.csv` |
| 5.9.6 | E13 五基线对比 | `真实数据五基线对比.md` | **手写**（协议 + 两口径全表 + 五点结论） | `realdata_official/`、`realdata_early/` |
| 5.9.7 | E14 多种子重复 | `真实数据多种子重复.md` | **手写**（含两次 OOM 的执行记录） | `multiseed/{official,early}/` |
| 5.10.4 | E9 调参搜索 | `调参实验-E9.md` | 自动（`scripts/tune.py`） | `optuna/trials.csv`、`optuna/best_params.json` |
| 5.10.4 | E9b 调参验证 | `调参验证-E9b.md` | 自动（`scripts/validate_tuning.py`） | `optuna/best_params.json` |
| 2.4 / 5.1 | 测试覆盖率 | `覆盖率报告.md` | **手写**（数据来自 `pytest --cov`） | 运行 `pytest --cov=mooc_dropout --cov-report=term-missing` 复现 |

手写报告共 4 份：`实验记录-合成数据.md`、`真实数据五基线对比.md`、`真实数据多种子重复.md`、
`覆盖率报告.md`。**其余 6 份 md 全部由脚本重写，要改内容先改脚本。**

## 二、原始数据与中间产物

| 路径 | 内容 | 谁写的 |
|---|---|---|
| `optuna/` | E9 的 trial 明细、最优参数、Optuna study 数据库 | `scripts/tune.py` |
| `realdata_official/`、`realdata_early/` | E13 两口径的 `metrics.json` 与 `compare/`（每模型一行） | `scripts/run_experiments.py --label-file` |
| `multiseed/official/`、`multiseed/early/` | E14 的 `raw.csv`（逐种子逐配置）、`summary.csv`、`paired.csv` | `scripts/multiseed_realdata.py` → `scripts/collect_multiseed.py` |
| `exp_*.csv` | 合成数据各实验的原始表（论文中所有合成数据数字的来源） | `run_experiments.py` / `run_extra_experiments.py` |
| `metrics*.json` | 单次训练的指标快照（`metrics.json` 与部署版模型对应，`metrics_uncalibrated_main.json` 对应未校准主结果） | `scripts/train.py` |
| `sample_request.json` | 接口演示用的样例请求体 | 手写 |
| `_deprecated_oof_misaligned_metrics.json` | **已废弃**：OOF 错位 bug 期间的指标，论文 5.9.3 作为「错误长什么样」的证据 | 修复前的旧运行 |

## 三、一键复现顺序

```bash
# 合成数据：E1~E5 → E6~E8 → E9 → E9b
python scripts/run_experiments.py --auto-synthetic --iterations 300
python scripts/run_extra_experiments.py
python scripts/tune.py --trials 30
python scripts/validate_tuning.py

# 真实数据：派生标签 → 训练 → E11/E12（只改决策，不重训）
python scripts/derive_early_dropout.py --data-dir data/raw/kdd2015
python scripts/train.py --data-dir data/raw/kdd2015
python scripts/cost_ratio_scan.py --auto
python scripts/capacity_analysis.py --auto

# 真实数据 E13（两种口径各一遍）→ 作图
python scripts/run_experiments.py --data-dir data/raw/kdd2015 --skip-sensitivity --skip-ablation
python scripts/run_experiments.py --data-dir data/raw/kdd2015 \
    --label-file data/processed/enrollment_early.csv --skip-sensitivity --skip-ablation
python scripts/plot_baseline_realdata.py

# 真实数据 E14（单进程跑，勿并行；每进程峰值约 3 GB）
python scripts/multiseed_realdata.py --tag official --seeds 42 7 2024 1234 3407
python scripts/multiseed_realdata.py --tag early \
    --label-file data/processed/enrollment_early.csv --seeds 42 7 2024 1234 3407
python scripts/collect_multiseed.py
python scripts/plot_multiseed_realdata.py
```

## 四、两条约定

1. **报告里的数字必须有 CSV/JSON 兜底**：手写报告只做解释，数字以对应的 csv/json 为准；
   两者不一致时，先怀疑报告过期（例如 E6 的种子数从 3 改到 5 之后，旧表会残留）。
2. **不要把 `artifacts/` 下的模型产物当成报告**：`artifacts/realdata_official/` 是答辩演示用的模型
   （校准版），`reports/realdata_official/` 是它的指标快照，两者不是一回事。
