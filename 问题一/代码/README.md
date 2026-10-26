# 第一问代码

主程序只有一个文件：[question1.py](question1.py)。它独立完成数据审计、字段核验、建模验证、七点评分、图表和模型导出；依赖版本列在同目录的 `requirements.txt`。

```powershell
# 只做清洗与字段核验（不训练）
& 'D:\conda\Anaconda3\python.exe' -X utf8 '问题一\代码\question1.py' --clean-only

# 完整流程
& 'D:\conda\Anaconda3\python.exe' -X utf8 '问题一\代码\question1.py'

# 新样本预测
& 'D:\conda\Anaconda3\python.exe' -X utf8 '问题一\代码\question1.py' --predict '新样本.csv'
```

`main()` 依次调用：

| 函数 | 作用 | 主要产物 |
|---|---|---|
| `run_cleaning()` | 缺失/重复/异常/尺度核验、来源逐值比对、评分卡字段对应与乙醛核验 | `数据表格/数据清洗/*`、`元数据/cleaning_summary.json`、`数据清洗报告.md` |
| `train()` | 外层5折＋内层3折逐属性选模、两种缺失遮蔽方案、区间实验、可靠性分档 | `数据表格/*.csv`、`元数据/summary.json` |
| `calibrate()` | 七点锚点拟合、OOF评分、敏感性、字段字典、统一评分模型 | `数据表格/七点评分/*`、`元数据/seven_point_summary.json` |
| `combine_models()` | 合并为单一可复用模型 | `模型/question1_model.joblib` |
| `plot_results()` | 逐属性R²、缺失情景曲线 | `绘图/`中的两张PNG |
| `paper_tables()` | 论文正文的2张表 | `数据表格/论文表格/*` |
| `write_report()` | 生成结果分析 | `问题分析/第一问结果分析.md` |

`--clean-only` 可单独重跑清洗核验：导出清洗特征/标签、缺失分布、重复检查、尺度字典、异常标记、来源逐值比对（含 `7.csv` 与作者 `Supplementary File S7`）、评分卡 52 项对应表与 `summary` 引用核验表。无需联网；若本地已有作者公开补充工作簿，还会核对作者 S1/S4/S7 与附件是否一致。

