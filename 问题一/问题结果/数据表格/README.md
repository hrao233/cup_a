# 第一问数据表格

连续预测、逐属性指标、两种缺失情景的实验表和论文用表格放在这里；七点评分表另放在 `七点评分/`。

| 文件 | 内容 |
|---|---|
| `oof_predictions_continuous.csv` | 250×50 连续感官属性的外层（OOF）预测 |
| `reference_targets_provided_scale.csv` | 对应的附件提供值 |
| `per_target_metrics.csv` | 逐属性 R²/MAE/RMSE/Spearman、基线 R²、`reliability` 可靠性档 |
| `model_comparison.csv` | 同一外层划分下各模型家族的宏平均指标 |
| `outer_fold_metrics.csv`、`outer_model_selection.csv`、`final_model_selection.csv`、`final_inner_cv_mse.csv` | 各折指标、逐属性选模结果与内层 MSE |
| `missingness_stress_raw.csv`、`missingness_stress_summary.csv` | 两种遮蔽方案（`random_all_features` 231 列 / `observed_missing_features` 9 列）× 两模型的缺失压力实验；`masked_cell_share_of_all` 给出该遮蔽量占全部 250×231 个单元格的比例，避免把"列内 30%"误读为全部单元格的 30% |
| `missingness_group_metrics.csv` | 按原始数据是否真的缺失分组的误差与样本数（组内比较，不代表因果） |
| `split_conformal_metrics.csv`、`split_conformal_test_predictions.csv` | 独立区间实验的逐属性覆盖率与测试预测 |
| `论文表格/` | 正文用表1（5 款代表样本 × 10 个主要维度）、支撑用完整表（12 款 × 30 核心项）与表2（模型评价，含固定等权树融合） |

`数据清洗/clean_features.csv` 与 `clean_targets.csv` 是对齐、规范后的模型输入。缺失测量仍以空值保存；补值及统计裁剪属于训练折内部的预处理。其余核验表保存在 `数据清洗/`：逐列尺度见 `feature_scales.csv`，评分卡 52 项对应见 `scorecard_item_mapping.csv`，模板引用错误见 `scorecard_summary_reference_errors.csv`，`7.csv` 乙醛列核验见 `alternative_acetaldehyde_candidates.csv`。

指标口径提醒：MAE/RMSE 均为附件连续标签尺度，不是 0—6 分误差；缺失实验表是折内指标再平均，与 250 个 OOF 预测汇总的宏平均 R² 分子分母不同，不能混用。
