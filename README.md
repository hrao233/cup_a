# A题第一问：从化学特征预测啤酒感官属性

仓库包含可独立运行的单文件程序、完整结果表、图表，以及可直接用于新样本预测的模型。

## 运行

需要 Python 及 `requirements.txt` 中的依赖。训练需要原始附件目录，至少包含 `1.csv`、`4.csv`、`3.xlsx`、`8.xlsx`：

```powershell
python -m pip install -r requirements.txt
python question1.py --data-dir "附件目录" --output output_rebuilt
```

完整训练会运行外层 5 折、内层 3 折交叉验证，并重新生成评分、报告、图表和模型。结果写入指定目录。仓库内已提供此前完整实验的结果；`output/verification.json` 记录了单文件整合时的核对范围。

不重新训练，直接预测新样本：

```powershell
python question1.py --predict "新样本.csv"
```

预测结果默认写入 `new_predictions`。输入特征名称及数值表示应与 `1.csv` 一致。

## 当前结果

- 250 个样本，231 个化学/理化特征，50 个感官预测目标。
- 五折外层验证的 50 项宏平均 R² 为 **0.2448**；30 项核心香气/风味宏平均 R² 为 **0.2010**。
- `output/seven_point/oof_core_aroma_flavour_grades_0_6.csv`：30 项核心属性的估计七点评分。
- `output/seven_point/oof_estimated_intensity_grades_0_6.csv`：49 项强度属性的估计七点评分，排除总体喜好度。
- `output/seven_point/scorecard_beer_1_to_12_estimated_grades.csv`：题目中 12 款啤酒的评分。
- `output/第一问结果.md`：方法、结果及限制；`output/figures`：图表。
- `output/question1_model.joblib`：已训练模型，可供预测及后续建模使用。

七点评分借鉴 QDP 和已有研究的七点量表形式。具体参照锚点属于经验假设，尚未通过人工强度评分校准；附件中的标准化标签无法唯一还原为原始人工 0—6 分，因此不报告原始等级准确率。

## 数据与验证说明

原始比赛附件没有放入仓库。模型训练时的插补、特征裁剪和标准化只在训练折中拟合。`output/verification.json` 明确记录：历史完整五折实验结果被复用来核对整合文件；整合版独立重训了第一折，并核对评分、模型序列化及新样本预测。

参考研究：Schreurs 等，2024，[*Predicting and improving complex beer flavor through machine learning*](https://doi.org/10.1038/s41467-024-46346-0)。
