## 趋势特征消融

| experiment | roc_auc | pr_auc | recall | precision | f1 | cost_per_student | cost_at_0.5 | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CatBoost full features | 0.7345 | 0.3626 | 0.7195 | 0.4041 | 0.5175 | 77.13 | 77.01 | 0.495 |
| CatBoost w/o temporal decay | 0.7345 | 0.3776 | 0.7622 | 0.3019 | 0.4325 | 82.6 | 79.68 | 0.3804 |
