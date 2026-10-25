"""Create the Chinese analysis report and standalone scientific figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from data import ROOT

OUT=ROOT/'results'
FIG=OUT/'figures'


def md_table(frame, columns=None):
    if columns is not None:
        frame=frame[columns]
    def value(v):
        return f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def main():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'per_target_metrics.csv',header=0)
    compare=pd.read_csv(OUT/'model_comparison.csv',header=0)
    stress=pd.read_csv(OUT/'missingness_stress_summary.csv',header=0)
    interval=pd.read_csv(OUT/'split_conformal_metrics.csv',header=0)
    target_names=metrics.set_index('target').CleanName_PanelFeedback.to_dict()
    arrays=np.load(OUT/'evaluation_arrays.npz')
    y,pred=arrays['y'],arrays['prediction']
    FIG.mkdir(exist_ok=True)
    installed={f.name for f in font_manager.fontManager.ttflist}
    preferred=[f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC','Arial Unicode MS'] if f in installed]
    plt.rcParams.update({'font.family':preferred+['DejaVu Sans'],'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc',
        'axes.facecolor':'white','axes.titleweight':'bold','font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    fig.suptitle('A题第一问 · 嵌套交叉验证结果',fontsize=20,fontweight='bold',color='#172554')
    plot=compare.sort_values('macro_r2')
    labels={'training_mean':'训练折均值基线','nested_selected':'按属性内层选择','hist_gradient_boost':'梯度提升树',
            'blend_tree_linear':'树模型＋岭回归等权融合'}
    axes[0,0].barh([labels.get(x,x) for x in plot.model],plot.macro_r2,
                  color=['#0f766e' if x=='nested_selected' else '#93c5fd' for x in plot.model])
    axes[0,0].axvline(0,color='#64748b',lw=.8)
    axes[0,0].set(title='50个感官属性：宏平均 R²',xlabel='外层折未见样本的预测表现')
    best=metrics.sort_values('r2',ascending=False).iloc[0]
    j=list(metrics.target).index(best.target)
    axes[0,1].scatter(y[:,j],pred[:,j],s=22,alpha=.65,c='#0f766e',edgecolors='white',linewidths=.3)
    lim=[min(y[:,j].min(),pred[:,j].min()),max(y[:,j].max(),pred[:,j].max())]
    axes[0,1].plot(lim,lim,'--',color='#64748b')
    axes[0,1].set(title=f'示例：{target_names[best.target]}，R²={best.r2:.3f}',
                  xlabel='附件提供的感官值',ylabel='外层预测值')
    for name,color,label in [('nested_selected','#0f766e','内层选择模型'),('mask_augmented_extra_trees','#c2410c','缺失增强树模型')]:
        part=stress[stress.method==name]
        axes[1,0].plot(part.additional_missing_rate*100,part.core_macro_r2,'o-',color=color,label=label)
    axes[1,0].set(title='成分额外随机缺失：30个核心香气/风味属性',xlabel='额外遮蔽比例（%）',ylabel='各外层折宏平均 R² 的均值')
    axes[1,0].legend(frameon=False);axes[1,0].grid(axis='y',alpha=.2)
    part=metrics.sort_values('r2',ascending=False).head(10).sort_values('r2')
    axes[1,1].barh(part.target,part.r2,color='#38bdf8')
    axes[1,1].set(title='预测表现最高的10个属性',xlabel='汇总外层预测 R²')
    fig.savefig(FIG/'question1_overview.png',dpi=170)
    fig.savefig(FIG/'question1_overview.svg')
    plt.close(fig)
    plot=metrics.sort_values('r2')
    fig,ax=plt.subplots(figsize=(11,13),layout='constrained')
    ax.barh(plot.target,plot.r2,color=np.where(plot.r2>0,'#0f766e','#c2410c'))
    ax.axvline(0,c='#64748b',lw=1)
    ax.set(title='全部50个属性的外层预测 R²（保留负值）',xlabel='R²；负值表示弱于按总体均值预测')
    fig.savefig(FIG/'all_target_r2.png',dpi=160);plt.close(fig)

    grades=pd.read_csv(OUT/'oof_relative_grades_0_6.csv',header=0,dtype={'beer':str,'beer_id':str})
    first12=grades[grades.beer.astype(int).between(1,12)].copy()
    assert len(first12)==12
    first12['beer_order']=first12.beer.astype(int)
    first12=first12.sort_values('beer_order').drop(columns='beer_order')
    first12.to_csv(OUT/'scorecard_beer_1_to_12_relative_grades.csv',index=False,encoding='utf-8-sig')
    top=metrics.sort_values('r2',ascending=False).head(10)
    weak=metrics.sort_values('r2').head(8)
    assert (OUT/'source_comparison.json').exists(), 'Run verify_sources.py to verify the provenance statements'
    lines=[
        '# A题第一问：分析、实现与首轮实验结果',
        '',
        '后续已采用QDP与来源研究的七点强度量表，当前评分方案见[七点强度评分方案与结果](seven_point/七点强度评分方案与结果.md)。本文保留首轮连续预测与历史相对分位数结果。',
        '',
        '本次完成了数据审计、50维感官预测、嵌套交叉验证、缺失压力测试、逐属性区间实验、相对0—6级输出及可复用模型。原始人工0—6绝对评分无法从现有公开数据唯一恢复；相对等级属于显式建模假设，不能作为原始评分已还原的结论。',
        '',
        '## 1. 第一问如何理解',
        '',
        '输入为1.csv的231个化学/理化特征；训练目标为4.csv的50个感官属性。3.xlsx定义评分项目和表单结构。题面“约40维”与附件存在差异，本次首先覆盖可明确对应的全部50维，并把Aroma与Flavour各15维、合计30维作为核心指标单独评价。',
        '',
        '3.xlsx实际有12张beer评分卡及summary。评分卡里的0是未填写输入时公式计算的默认值；不是12个已完成品评的训练样本。summary的50个属性可与4.csv逐项对应，需处理A_ester_fruity/A_esters_fruity、F_ester_fruity/F_esters_fruity、4vg/X4vg等命名差异。卡片另列的Acetaldehyde没有对应的独立Aroma/Flavour训练标签，本次不伪造该项目的分数。',
        '',
        '## 2. 数据审计与预处理',
        '',
        f'- 样本：{audit["samples"]}；化学特征：{audit["features"]}；感官属性：{audit["targets"]}。',
        f'- 化学缺失值：{audit["missing_x"]}/{250*231}，约{100*audit["missing_x"]/(250*231):.3f}%；感官标签无缺失。',
        '- 按beer编号连接；beer_id部分已被科学计数法舍入，不能作为精确连接键。先后样本为beer 10和beer 157；不是按编号排序。',
        '- 与8.xlsx的70个原始校准指标核验：37列符合log10表示，33列与原始值一致，其余161列的绝对浓度尺度尚未核实。pH和ethanol等在本附件中也符合log10表示。',
        '- 建模保留1.csv已提供的数值表示，避免再次全表取对数；中位数插补与标准化仅在训练折拟合。低值及负值不会被直接删除或截为0。',
        '- 不把风格、样本编号、论坛评论、附带SHAP/重要性结果作为化学预测模型的输入。',
        '',
        '## 3. 原始评分尺度为什么尚未恢复',
        '',
        '原始论文说明，感官评分按品评员分别中心化和标准差缩放；附件是处理后的汇总值。即便知道一个全局均值和标准差，也未必能反推每位品评员的原始评分与汇总过程。',
        '',
        '作者Zenodo公开补充工作簿S1、S4的化学/感官数值列与本题1.csv、4.csv一致（最大浮点差异分别约8.9e-16和1.1e-16），按beer核对行；存在舍入的beer_id不参与数值一致性比较。其质控脚本引用Raw_Sensory_data_Trained_panel.csv，但该逐人原始表不在公开文件清单中。本次没有找到将标准化汇总分唯一还原成0—6绝对等级的锚点。',
        '',
        '原始研究：[Nature Communications论文](https://doi.org/10.1038/s41467-024-46346-0)；[作者公开补充数据](https://zenodo.org/records/10653704)。下载来源与SHA-256见external_sources/manifest.json。',
        '',
        '因此分开交付：①可验证的附件连续尺度预测；②假设下的0—6相对等级。后者的0表示相对低组，不能解释为“无此风味”。',
        '不可辨识性的一个简单例子：若原始评分在不越界的范围内整体加一个常数，同时改变其均值，标准化后的值保持不变。因此没有原始尺度锚点时，仅凭标准化值不能确定原来的绝对分数。',
        '',
        '## 4. 模型与验证设计',
        '',
        '- 外层5折评价未见样本预测，内层3折按属性选择候选模型；固定种子20261024。每个样本只作为一次外层测试样本。',
        '- 候选为3个正则强度的岭回归、2个潜变量数的PLS、2种叶节点限制的多目标ExtraTrees、逐属性ExtraTrees、逐属性直方图梯度提升树、两种固定等权融合及训练折均值基线。',
        '- 首轮未约束的线性方法在极端特征样本上出现严重外推。修订版在线性与PLS分支加入训练折1%—99%分位数截尾，树模型保留原特征。首轮记录保存在experiments/initial_unbounded_linear，未隐藏失败结果。',
        '- 各模型的特征及目标标准化在训练数据中拟合，按属性比较内层均方误差。测试标签不参与插补、选参、模型选择和等级映射。',
        '- 不使用含自身标签训练得到的拟合值当作泛化成绩。报告指标来自外层预测。最终可复用模型随后才在全部250个样本上重新拟合。',
        '- 外层为随机KFold；本次检验同一总体的新样本表现，没有证明新地区、新工艺或未见风格下的泛化。候选流程根据首轮诊断做过修订，结果属于本数据上的开发验证，最终仍应有独立外部验证。',
        '',
        '数学上，建立X∈R^(250×231)、Y∈R^(250×50)。对第j个感官属性，使用训练折估计填补器I、截尾器C（仅线性分支）、标准化器S，再学习f_j，使训练感官值与预测值的平方误差尽量小。正则强度、潜变量数和模型家族由内层验证决定。最终输出ŷ_j=f_j(S(C(I(x))))，再通过独立说明的映射G_j得到临时相对等级。',
        '',
        '## 5. 连续感官预测结果',
        '',
        f'- 全50维宏平均R²：**{summary["macro_r2"]:.4f}**。',
        f'- 核心30维Aroma/Flavour宏平均R²：**{summary["core_macro_r2"]:.4f}**。',
        f'- 汇总RMSE：{summary["rmse_provided_scale"]:.4f}；MAE：{summary["mae_provided_scale"]:.4f}。两者均在附件提供的感官尺度上，不是原始0—6分误差。',
        f'- 相对训练折均值基线的汇总RMSE下降：{summary["rmse_reduction_vs_training_mean"]*100:.2f}%。',
        f'- 全50维R²固定OOF预测的样本bootstrap 95%区间：{summary["bootstrap_macro_r2_95_interval_fixed_oof"]}。该区间不包含重训及重新划分训练集的全部不确定性。',
        f'- R²大于0的属性：{int((metrics.r2>0).sum())}/50；不大于0的属性：{int((metrics.r2<=0).sum())}/50。不同属性可预测性差异明显。',
        '',
        md_table(compare.sort_values('macro_r2',ascending=False),['model','macro_r2','core_macro_r2','rmse_provided_scale']),
        '',
        '各固定候选模型使用相同外层测试划分。按属性选择本身也在外层评价，因此不能保证它在这次有限样本中胜过所有固定模型；这里保留全部结果。',
        '',
        '预测表现较好的属性：',
        '',
        md_table(top,['target','CleanName_PanelFeedback','r2','rmse']),
        '',
        '表现较弱的属性（保留负R²）：',
        '',
        md_table(weak,['target','CleanName_PanelFeedback','r2','rmse']),
        '',
        'R²约0代表接近均值预测，负值表示弱于均值预测。对于弱维度，不能仅因输出了一个数字，就解释为模型能够可靠识别这种气味。',
        '',
        '![第一问实验概览](figures/question1_overview.png)',
        '',
        '## 6. 数据缺失条件下的表现',
        '',
        '在每个外层测试折原有缺失基础上，额外随机遮蔽10%、20%、30%的特征单元，每档重复5次。填补器和模型只在训练折拟合，测试时不使用感官真值填补。重复先在折内平均，再汇总各折。0%仅指没有新增遮蔽，仍包含原始缺失。',
        '',
        '另设一个训练时缺失增强的树模型对照：在训练样本的复制版本上分别遮蔽10%、20%、30%特征；所有复制版本都留在训练折内，避免同一啤酒同时进入训练和测试。',
        '',
        md_table(stress,['method','additional_missing_rate','macro_r2','core_macro_r2','rmse','fold_sd_r2']),
        '',
        '此实验覆盖随机测量缺失（MCAR），没有证明低于检测限、仪器选择性缺测、整个化合物类别缺失等机制下的可靠性。',
        '本表R²是各外层折R²的均值；前文主指标是在250个外层预测上汇总计算。两者的方差分母不同，因此0%新增缺失一行不必与主指标完全相同。',
        '',
        '## 7. 0—6相对等级及尺度敏感性',
        '',
        '主映射对每个属性使用训练折标签的经验分布F，将预测值映射为min(6,floor(7F))；重复值采用中秩。每一等级代表训练分布中的一个相对区间，不是原始品评强度的语义刻度。',
        '',
        '另给一组显式假设：训练均值对应3分、低于均值2个标准差对应0分、高于2个标准差对应6分，中间线性转换后取最近整数并截断至0—6。',
        '',
        f'两种映射的等级不一致比例为{summary["relative_grade_mapping_disagreement_fraction"]*100:.2f}%，平均等级差为{summary["grade_mae_between_mapping_assumptions"]:.3f}。这是对尺度假设的敏感性，不是预测准确率。',
        '',
        '对250个样本均交付连续OOF预测与相对等级；另按Beer# 1—12导出评分卡对应的12个样本，明确它们的相对等级性质。未把相对等级回填原始评分卡以冒充真实人工打分。',
        '',
        '## 8. 独立区间预测实验',
        '',
        '固定拆分150训练、50校准、50测试；所有选模仅用150个训练样本，50校准样本仅确定逐属性绝对残差分位数，最后50个测试样本仅评价覆盖率。名义逐属性覆盖率90%，有限样本分位数使用ceil((n_cal+1)*0.9)阶统计量。',
        '',
        f'测试集50个属性的平均覆盖率：{summary["split_conformal_90_mean_coverage"]*100:.2f}%；核心30属性平均覆盖率：{summary["split_conformal_90_core_mean_coverage"]*100:.2f}%。完整逐属性覆盖率及宽度在split_conformal_metrics.csv。平均覆盖率不能替代每个属性的实际表现。',
        '',
        '区间保证依赖训练/校准/测试样本可交换等条件，不是50维同时覆盖保证，也不直接适用于第三问明显偏离训练分布的生成样本。此模型只用150个样本拟合，其点预测与最终250样本模型不同，不能混用两套点预测和区间半径。',
        '',
        '## 9. 本轮结论与尚未解决的部分',
        '',
        '第一问已形成可复现的连续感官预测方案，并量化不同属性、不同缺失率下的性能。不能将本轮结果描述成“全部感官维度都能准确预测”，也不能把相对分位数等级描述成原始人工评分恢复。',
        '',
        '若要完成题面0—6绝对等级的严格校准，仍需原始逐人评分及标准化/汇总规则，或官方认可的原始分数锚点。仅凭现有标准化汇总表存在不可辨识性。新增真实校准数据后可在预测模型之后接入校准器，并用独立样本评价等级误差。',
        '',
        '第二、三问可以复用models/sensory_model.joblib的连续感官模型；同时应参考本轮每个属性的误差和训练分布范围，不把低可预测性属性的输出当作确定事实。',
        '',
        '## 10. 复现与核验',
        '',
        f'运行耗时：{summary["elapsed_seconds"]:.1f}秒（不含数据溯源、图表与报告）。验证：{summary["verification"]}。',
        '',
        '数据字典、量纲核验、缺失统计、输入SHA-256、外层各折样本编号、逐属性选模、模型比较、压力测试原始记录和区间测试预测均随项目保存。运行命令及推理方法见项目README.md。',
    ]
    (OUT/'第一问分析与实验结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report and figures saved:',OUT)


if __name__=='__main__':
    main()
