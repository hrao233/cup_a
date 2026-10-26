"""A题第一问：单文件完整程序。

训练与输出：python "问题一/代码/question1.py"
指定附件：  python "问题一/代码/question1.py" --data-dir "附件文件夹"
新样本预测：python "问题一/代码/question1.py" --predict "新样本.csv"
仅数据清洗：python "问题一/代码/question1.py" --clean-only

包含：数据审计、训练折内预处理、5×3嵌套交叉验证、多模型比较、
缺失压力实验、独立区间实验、QDP七点估计评分、图表、报告及模型保存。
只依赖原始附件和常用第三方包，不导入本项目其他Python文件，不需要联网。

注意：七点评分的经验参照锚点是建模假设，并非已校准的历史人工评分。
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv as csvlib
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import openpyxl
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr, fisher_exact
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# A stable module name lets the same file save/load its custom estimator classes,
# whether executed as a script or imported for question 2/3.
sys.modules.setdefault('question1', sys.modules[__name__])
ROOT = Path(__file__).resolve().parent
if ROOT.name == '代码' and ROOT.parent.name == '问题一':
    ROOT = ROOT.parent.parent
WORKSPACE = ROOT.parent
DATA_DIR = None
META = ['beer', 'beer_id', 'tasting_category_fine']
SEED = 20261024
QUESTION_DIR = ROOT / '问题一'
OUT = QUESTION_DIR / '问题结果'
TABLES = OUT / '数据表格'
INFO = OUT / '元数据'
MODEL_DIR = OUT / '模型'
SCORE_OUT = TABLES / '七点评分'
FIG = OUT / '绘图'

# Data processing and estimator definitions follow below.


def attachment_dir():
    if DATA_DIR is not None:
        return DATA_DIR
    paths = [p.parent for p in WORKSPACE.rglob('1.csv')
             if (p.parent / '3.xlsx').exists() and (p.parent / '8.xlsx').exists()]
    assert len(paths) == 1, paths
    return paths[0]


def sheet_frame(workbook, sheet):
    """Explicit first-row header, no pandas implicit header inference."""
    ws = workbook[sheet]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame()
    headers = list(rows[0])
    frame = pd.DataFrame(rows[1:], columns=headers)
    assert len(frame) == ws.max_row - 1
    return frame


def read_csv_checked(path, expected_shape):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        headers = [name.strip() for name in next(csvlib.reader(stream))]
    if not all(headers) or len(headers) != len(set(headers)):
        raise ValueError(f'{path.name}: empty or duplicate column names')
    frame = pd.read_csv(path, header=0, names=headers, encoding='utf-8-sig', dtype={c: 'string' for c in META})
    assert frame.shape == expected_shape, (path.name, frame.shape, expected_shape)
    for column in META:
        frame[column] = frame[column].astype('string').str.strip()
        if frame[column].isna().any() or frame[column].eq('').any():
            raise ValueError(f'{path.name}: missing metadata in {column}')
    assert frame['beer'].is_unique
    # Beer IDs stored in rounded scientific notation are not valid join keys.
    return frame


def feature_inventory(features, X, calibrated, thresholds):
    """Separate verified transformations from uncalibrated supplied signals.

    The paper's perception thresholds are NOT analytical detection limits.
    Matching the median only identifies a dominant representation; every
    calibration cell is compared separately in run_cleaning().
    """
    dictionary = thresholds.dropna(subset=['field_name_R']).set_index('field_name_R')
    if not dictionary.index.is_unique or not set(features) <= set(dictionary.index):
        raise ValueError('Chemical dictionary is ambiguous or incomplete')
    rows = []
    for j, name in enumerate(features):
        source = dictionary.loc[name]
        is_calibrated = name in calibrated.columns
        aggregate = name in {'esters_sum', 'aroma_hops_sum', 'acids_sum', 'sulfur_sum'}
        row = dict(feature=name, chemical_name=source.chemical_name,
                   measurement_kind='calibrated' if is_calibrated else 'aggregate_signal' if aggregate else 'uncalibrated_signal',
                   source_unit=source.unit,
                   interpretation_unit=source.unit if is_calibrated else 'supplied processed relative signal (A.U.)',
                   representation='aggregate_signal_as_supplied' if aggregate else 'relative_signal_as_supplied',
                   checked_pairs=0, median_log10_error=np.nan, median_raw_error=np.nan,
                   calibrated_reference_available=is_calibrated,
                   positive_reference_transform_verified=False,
                   sensory_threshold_median_not_lod=source.threshold_median,
                   action='preserve supplied values; impute only within training fold')
        if is_calibrated:
            raw = pd.to_numeric(calibrated[name], errors='coerce').to_numpy(float)
            provided = X[:,j]
            keep = np.isfinite(raw) & np.isfinite(provided) & (raw > 0)
            e_log = float(np.median(np.abs(np.log10(raw[keep])-provided[keep])))
            e_raw = float(np.median(np.abs(raw[keep]-provided[keep])))
            mode = 'log10' if e_log < 1e-7 else 'raw' if e_raw < 1e-7 else 'unresolved_calibration'
            row.update(representation=mode, checked_pairs=int(keep.sum()),
                       median_log10_error=e_log, median_raw_error=e_raw,
                       positive_reference_transform_verified=bool(np.allclose(
                           np.log10(raw[keep]) if mode == 'log10' else raw[keep],
                           provided[keep], rtol=1e-7, atol=1e-7)))
        rows.append(row)
    return pd.DataFrame(rows)


def load_data(out=None):
    source = attachment_dir()
    chem = read_csv_checked(source / '1.csv', (250, 234))
    sensory = read_csv_checked(source / '4.csv', (250, 53))
    assert set(chem.beer) == set(sensory.beer)
    sensory = sensory.set_index('beer').loc[chem.beer].reset_index()
    assert np.array_equal(chem.tasting_category_fine, sensory.tasting_category_fine)
    features = list(chem.columns[3:])
    targets = list(sensory.columns[3:])
    X = chem[features].apply(pd.to_numeric, errors='raise').astype(float).to_numpy()
    Y = sensory[targets].apply(pd.to_numeric, errors='raise').astype(float).to_numpy()
    assert np.isfinite(Y).all()
    assert not np.isinf(X).any()
    assert chem.beer.iloc[0] == '10' and chem.beer.iloc[-1] == '157'
    assert len(chem.beer.unique()) == 250

    wb8 = openpyxl.load_workbook(source / '8.xlsx', read_only=True, data_only=True)
    names = sheet_frame(wb8, 'Sensory names')
    names = names[names['dataset'] == 'Trained panel'].set_index('R_name')
    mapping = pd.DataFrame({'target': targets})
    for col in ['Category_Paper', 'CleanName_PaperLong', 'CleanName_PanelFeedback', 'DataType']:
        mapping[col] = [names.loc[t, col] if t in names.index else None for t in targets]
    mapping['scope'] = ['core_aroma_flavour' if t.startswith(('A_', 'F_')) else 'additional' for t in targets]
    assert (mapping.scope == 'core_aroma_flavour').sum() == 30

    wb3 = openpyxl.load_workbook(source / '3.xlsx', read_only=True, data_only=False)
    wb3_values = openpyxl.load_workbook(source / '3.xlsx', read_only=True, data_only=True)
    summary_rows = list(wb3['summary'].iter_rows(values_only=True))
    alias = {'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity', '4vg': 'X4vg'}
    summary_names = [alias.get(v, v) for v in summary_rows[0][1:] if v != 'comments']
    assert set(summary_names) == set(targets), (set(summary_names) - set(targets), set(targets) - set(summary_names))
    cards = [s for s in wb3.sheetnames if s.startswith('beer_')]
    score_cells = [c.value for row in wb3['beer_1'].iter_rows() for c in row if c.data_type == 'f']
    cached = [wb3_values['beer_1'][c.coordinate].value for row in wb3['beer_1'].iter_rows()
              for c in row if c.data_type == 'f']
    assert all(v == 0 for v in cached), 'Unexpected non-default scorecard entries: inspect before training'

    thresholds = sheet_frame(wb8, 'Sensory thresholds')
    calibrated = sheet_frame(wb8, 'Calibrated chemical dataset')
    calibrated['beer'] = calibrated.beer.astype(str)
    calibrated = calibrated.set_index('beer').loc[chem.beer]
    scales = feature_inventory(features, X, calibrated, thresholds)
    wb8.close(); wb3.close(); wb3_values.close()

    manifest = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(source.iterdir()) if p.is_file()]
    audit = dict(samples=250, features=len(features), targets=len(targets), core_targets=30,
                 missing_x=int(np.isnan(X).sum()), missing_y=int(np.isnan(Y).sum()),
                 first_beer=chem.beer.iloc[0], last_beer=chem.beer.iloc[-1],
                 card_count=len(cards), card_formula_count=len(score_cells),
                 card_formula_cached_values=sorted(set(cached)),
                 style_counts=chem.tasting_category_fine.value_counts().to_dict(),
                 scale_counts=scales.representation.value_counts().to_dict(),
                 supplied_score_scale='panelist-standardized, original 0-6 anchors unavailable',
                 input_manifest=manifest)
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(out / 'target_dictionary.csv', index=False, encoding='utf-8-sig')
        scales.to_csv(out / 'feature_scales.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame({'feature': features, 'missing_count': np.isnan(X).sum(axis=0),
                      'missing_fraction': np.isnan(X).mean(axis=0)}).to_csv(out / 'missingness.csv', index=False, encoding='utf-8-sig')
        INFO.mkdir(parents=True, exist_ok=True)
        (INFO / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return chem[META].copy(), X, Y, features, targets, mapping, audit


def run_cleaning():
    """Audit and export model-ready source values, without global imputation.

    Cleaning preserves the supplied predictor/target arrays. Missing-value
    imputation and statistical clipping remain fitted inside each training fold.
    """
    for directory in [OUT, TABLES, INFO, FIG]:
        directory.mkdir(parents=True, exist_ok=True)
    destination = TABLES/'数据清洗'
    destination.mkdir(exist_ok=True)

    def export(frame, name):
        frame.to_csv(destination/name, index=False, encoding='utf-8-sig')

    meta, X, Y, features, targets, mapping, base_audit = load_data(TABLES)
    source = attachment_dir()
    chemistry = pd.DataFrame(X, columns=features)
    sensory = pd.DataFrame(Y, columns=targets)
    book = openpyxl.load_workbook(source/'8.xlsx', read_only=True, data_only=True)
    reference = sheet_frame(book, 'Calibrated chemical dataset')
    reference.beer = reference.beer.astype(str)
    reference = reference.set_index('beer').loc[meta.beer]
    thresholds = sheet_frame(book, 'Sensory thresholds')
    inventory = feature_inventory(features, X, reference, thresholds)
    book.close()
    inventory.to_csv(TABLES/'feature_scales.csv', index=False, encoding='utf-8-sig')
    # The precise IDs in the chemistry table are independently present in 8.xlsx.
    if not np.array_equal(meta.beer_id.astype(str).to_numpy(), reference.beer_id.astype(str).to_numpy()):
        raise ValueError('Chemistry IDs disagree with the independent reference table')
    original_y = read_csv_checked(source/'4.csv', (250,53)).set_index('beer').loc[meta.beer]
    repairs = []
    for i in range(len(meta)):
        if str(original_y.beer_id.iloc[i]) != str(meta.beer_id.iloc[i]):
            repairs.append(dict(beer=meta.beer.iloc[i], field='beer_id',
                original=original_y.beer_id.iloc[i], cleaned=meta.beer_id.iloc[i],
                decision='use exact chemistry/reference ID after joining on beer; predictor values unchanged'))
    export(pd.DataFrame(repairs, columns=['beer','field','original','cleaned','decision']), 'metadata_repairs.csv')
    export(pd.DataFrame([
        {'source_field':a,'canonical_field':b,'context':'scorecard summary; no target values altered'}
        for a,b in {'A_ester_fruity':'A_esters_fruity','F_ester_fruity':'F_esters_fruity','4vg':'X4vg'}.items()
    ]), 'attribute_name_mapping.csv')
    export(pd.concat([meta,chemistry],axis=1), 'clean_features.csv')
    export(pd.concat([meta,sensory],axis=1), 'clean_targets.csv')
    missing = np.isnan(X)
    export(pd.concat([meta,pd.DataFrame(missing.astype(int),columns=features)],axis=1), 'missing_mask.csv')
    rows = meta.copy()
    rows['missing_count'] = missing.sum(axis=1)
    rows['missing_fraction'] = missing.mean(axis=1)
    rows['missing_features'] = [';'.join(np.array(features)[mask]) for mask in missing]
    export(rows, 'missingness_by_sample.csv')
    groups = []
    for style in sorted(meta.tasting_category_fine.unique()):
        select = meta.tasting_category_fine.eq(style).to_numpy()
        groups.append(dict(style=style, samples=int(select.sum()), affected_samples=int(missing[select].any(axis=1).sum()),
                           missing_cells=int(missing[select].sum()), missing_fraction=float(missing[select].mean())))
    export(pd.DataFrame(groups), 'missingness_by_style.csv')
    patterns = pd.Series(rows.missing_features).value_counts().rename_axis('missing_features').reset_index(name='samples')
    export(patterns, 'missingness_patterns.csv')
    tests = []
    for j in np.flatnonzero(missing.any(axis=0)):
        for style in sorted(meta.tasting_category_fine.unique()):
            group = meta.tasting_category_fine.eq(style).to_numpy()
            a=int(missing[group,j].sum());b=int(group.sum())-a
            c=int(missing[~group,j].sum());d=int((~group).sum())-c
            tests.append(dict(feature=features[j],style=style,style_missing=a,style_total=a+b,
                              others_missing=c,others_total=c+d,p_value=float(fisher_exact([[a,b],[c,d]]).pvalue)))
    association = pd.DataFrame(tests)
    order = np.argsort(association.p_value.to_numpy())
    adjusted = association.p_value.to_numpy()[order]*len(order)/np.arange(1,len(order)+1)
    association.loc[order,'bh_adjusted_p'] = np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1])
    association['decision'] = np.where(association.bh_adjusted_p<.05,'association detected; missingness not assumed random',
                                      'no association detected; does not establish MCAR')
    export(association, 'missingness_style_tests.csv')

    duplicate_records = []
    for kind, frame, identifiers in [('chemical_rows',chemistry,meta.beer),('sensory_rows',sensory,meta.beer),
                                      ('chemical_columns',chemistry.T,pd.Series(features))]:
        repeated = frame.duplicated(keep=False).to_numpy()
        for pos in np.flatnonzero(repeated):
            duplicate_records.append(dict(kind=kind,identifier=str(identifiers.iloc[pos]),
                                           decision='review as possible replicate; do not silently delete'))
    export(pd.DataFrame(duplicate_records,columns=['kind','identifier','decision']), 'duplicate_audit.csv')
    const = chemistry.nunique(dropna=True)
    export(pd.DataFrame({'feature':features,'unique_observed_values':const.to_numpy(),
                        'constant_or_empty':const.to_numpy()<=1}), 'constant_feature_audit.csv')
    spread = np.nanquantile(X,.75,axis=0)-np.nanquantile(X,.25,axis=0)
    spread = np.where(spread>1e-12,spread,np.nanstd(X,axis=0))
    spread = np.where(spread>1e-12,spread,1.)
    pairs = []
    for i in range(len(X)):
        for k in range(i):
            observed = np.isfinite(X[i]) & np.isfinite(X[k])
            delta = np.abs((X[i,observed]-X[k,observed])/spread[observed])
            pairs.append(dict(beer_a=meta.beer.iloc[k],beer_b=meta.beer.iloc[i],shared_features=int(observed.sum()),
                              rms_iqr_scaled_difference=float(np.sqrt(np.mean(delta**2))),
                              max_iqr_scaled_difference=float(delta.max())))
    near = pd.DataFrame(pairs)
    near['near_identical'] = (near.shared_features>=.9*len(features)) & (near.max_iqr_scaled_difference<=1e-6)
    export(near.sort_values('rms_iqr_scaled_difference').head(20), 'nearest_sample_pairs.csv')

    profiles=[]; flagged=[]; row_flags=np.zeros(len(X),dtype=int)
    for dataset, values, names in [('chemical',X,features),('sensory',Y,targets)]:
        q1,median,q3 = np.nanquantile(values,[.25,.5,.75],axis=0)
        iqr=q3-q1;mad=np.nanmedian(np.abs(values-median),axis=0)
        iqr_flag=((values<q1-1.5*iqr)|(values>q3+1.5*iqr)) & (iqr>1e-12)
        mad_flag=(np.abs(values-median)>6*1.4826*mad) & (mad>1e-12)
        combined=iqr_flag|mad_flag
        if dataset=='chemical':row_flags=combined.sum(axis=1)
        for j,name in enumerate(names):
            observed=values[np.isfinite(values[:,j]),j]
            profiles.append(dict(dataset=dataset,feature=name,observed_count=len(observed),
                missing_count=int(np.isnan(values[:,j]).sum()),minimum=float(observed.min()),
                q1=float(q1[j]),median=float(median[j]),q3=float(q3[j]),maximum=float(observed.max()),
                iqr=float(iqr[j]),mad=float(mad[j]),unique_values=int(len(np.unique(observed))),
                minimum_frequency=int(np.sum(observed==observed.min())),negative_count=int(np.sum(observed<0)),
                iqr_flags=int(iqr_flag[:,j].sum()),mad_flags=int(mad_flag[:,j].sum()),
                decision='retain statistical extremes; neither rule establishes a measurement error'))
        for i,j in zip(*np.where(combined)):
            flagged.append(dict(dataset=dataset,beer=meta.beer.iloc[i],feature=names[j],value=float(values[i,j]),
                iqr_flag=bool(iqr_flag[i,j]),mad_6sigma_flag=bool(mad_flag[i,j]),
                robust_mad_score=float(abs(values[i,j]-median[j])/(1.4826*mad[j])) if mad[j]>1e-12 else np.nan,
                decision='retain supplied value; no source evidence that this is a recording error'))
    export(pd.DataFrame(profiles), 'distribution_profile.csv')

    comparisons=[]; per_feature=[]; physical=[]
    for j,name in enumerate(features):
        if name not in reference:continue
        raw=pd.to_numeric(reference[name],errors='coerce').to_numpy(float);provided=X[:,j]
        mode=inventory.set_index('feature').loc[name,'representation']
        positive=raw[np.isfinite(raw)&(raw>0)]
        half_min=float(positive.min()/2) if len(positive) else np.nan
        counts={}
        for i,(a,b) in enumerate(zip(raw,provided)):
            expected=np.nan
            if not np.isfinite(a):
                state='reference_missing_supplied_present' if np.isfinite(b) else 'both_missing'
            elif not np.isfinite(b):
                state='source_zero_to_missing' if a==0 else 'supplied_missing_reference_present'
            elif mode=='log10' and a<=0:
                expected=float(np.log10(half_min))
                state='nonpositive_reference_half_min_floor' if np.isclose(b,expected,rtol=1e-7,atol=1e-7) else 'reference_disagreement'
            else:
                expected=float(np.log10(a)) if mode=='log10' else float(a)
                state='matched' if np.isclose(b,expected,rtol=1e-7,atol=1e-7) else 'reference_disagreement'
            counts[state]=counts.get(state,0)+1
            if state not in {'matched','both_missing'}:
                comparisons.append(dict(beer=meta.beer.iloc[i],feature=name,representation=mode,
                    reference_value=a,supplied_value=b,expected_from_reference=expected,status=state,
                    decision='keep competition/S1 value; do not overwrite from a discordant reference or invent a detection limit'))
            invalid=np.isfinite(a) and (a<0 or (name=='pH' and a>14) or (name=='ethanol..v.v.' and a>100))
            if invalid:
                physical.append(dict(beer=meta.beer.iloc[i],feature=name,reference_value=a,supplied_value=b,
                    status='invalid physical reference value',
                    decision='reference is not a reliable replacement; processed source uses floor if confirmed in cell audit'))
        per_feature.append(dict(feature=name,representation=mode,comparison_atol=1e-7,comparison_rtol=1e-7,
                                reference_positive_min_half=half_min,**counts))
    comparison_frame=pd.DataFrame(comparisons)
    export(comparison_frame, 'calibration_cell_audit.csv')
    export(pd.DataFrame(per_feature).fillna(0), 'calibration_feature_audit.csv')
    export(pd.DataFrame(physical,columns=['beer','feature','reference_value','supplied_value','status','decision']), 'physical_reference_flags.csv')

    scorecards=[]
    formulas=openpyxl.load_workbook(source/'3.xlsx',read_only=True,data_only=False)
    cache=openpyxl.load_workbook(source/'3.xlsx',read_only=True,data_only=True)
    for name in formulas.sheetnames:
        if not name.startswith('beer_'):continue
        cells=[cell for row in formulas[name].iter_rows() for cell in row if cell.data_type=='f']
        cached=[cache[name][cell.coordinate].value for cell in cells]
        scorecards.append(dict(sheet=name,formula_cells=len(cells),default_zero_cells=sum(v==0 for v in cached),
                               decision='blank template; never use cached zeros as sensory labels'))
        if not cells or any(v!=0 for v in cached):
            raise ValueError(f'Scorecard {name} no longer matches the audited blank template')
    formulas.close();cache.close()
    export(pd.DataFrame(scorecards), 'scorecard_audit.csv')

    # Independent published workbook confirms that apparent calibration conflicts
    # already exist upstream; no attachment values are repaired from guesses.
    provenance=[]
    paper_workbook=QUESTION_DIR/'问题分析'/'研究来源材料'/'Supplemental Files and Figure source files.xlsx'
    if paper_workbook.exists():
        original=openpyxl.load_workbook(paper_workbook,read_only=True,data_only=True)
        for sheet,expected,columns in [('Supplementary File S1',X,features),('Supplementary File S4',Y,targets)]:
            frame=sheet_frame(original,sheet);frame.beer=frame.beer.astype(str)
            matrix=frame.set_index('beer').loc[meta.beer,columns].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            identical_missing=np.array_equal(np.isnan(matrix),np.isnan(expected))
            agrees=bool(np.allclose(matrix,expected,rtol=1e-7,atol=1e-7,equal_nan=True))
            if not agrees or not identical_missing:raise ValueError(f'Competition input differs from author source {sheet}')
            provenance.append(dict(source=sheet,status='matches published source including missing locations',
                max_absolute_difference=float(np.nanmax(np.abs(matrix-expected))),
                url='https://zenodo.org/records/10653704',sha256=hashlib.sha256(paper_workbook.read_bytes()).hexdigest()))
        author_reference=sheet_frame(original,'Calibrated chemical dataset')
        author_reference.beer=author_reference.beer.astype(str)
        columns=[name for name in features if name in reference]
        authored=author_reference.set_index('beer').loc[meta.beer,columns].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        attached=reference[columns].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        if not np.allclose(authored,attached,rtol=1e-7,atol=1e-7,equal_nan=True):
            raise ValueError('Attached calibration table differs from the published reference table')
        provenance.append(dict(source='Calibrated chemical dataset',status='also matches published source; cross-sheet conflicts are upstream',
            max_absolute_difference=float(np.nanmax(np.abs(authored-attached))),
            url='https://zenodo.org/records/10653704',sha256=hashlib.sha256(paper_workbook.read_bytes()).hexdigest()))
        original.close()
    export(pd.DataFrame(provenance,columns=['source','status','max_absolute_difference','url','sha256']), 'published_source_comparison.csv')
    outliers=pd.DataFrame(flagged,columns=['dataset','beer','feature','value','iqr_flag','mad_6sigma_flag','robust_mad_score','decision'])
    reference_status={(row['beer'],row['feature']):row['status'] for row in comparisons}
    calibrated_names=set(inventory.loc[inventory.calibrated_reference_available,'feature'])
    outliers['reference_status']=[
        'standardized_sensory' if row.dataset=='sensory' else
        reference_status.get((row.beer,row.feature),'matched_reference') if row.feature in calibrated_names else
        'no_absolute_calibration_reference'
        for row in outliers.itertuples()]
    outliers['matches_published_processed_source']=bool(provenance)
    export(outliers, 'statistical_outlier_flags.csv')
    extreme=outliers.loc[outliers.dataset=='chemical'].sort_values('robust_mad_score',ascending=False).head(20)
    export(extreme, 'extreme_value_review.csv')
    # Retain actual input bytes and prove that canonical exports did not alter measurements.
    check_x=pd.read_csv(destination/'clean_features.csv',header=0)[features].to_numpy(float)
    check_y=pd.read_csv(destination/'clean_targets.csv',header=0)[targets].to_numpy(float)
    np.testing.assert_allclose(check_x,X,rtol=1e-12,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(check_y,Y,rtol=1e-12,atol=1e-12)
    for entry in base_audit['input_manifest']:
        if hashlib.sha256((source/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError(f'Original attachment modified: {entry["file"]}')
    summary=dict(samples=len(meta),features=len(features),targets=len(targets),
        missing_cells=int(missing.sum()),missing_fraction=float(missing.mean()),
        missing_features=int(missing.any(axis=0).sum()),affected_samples=int(missing.any(axis=1).sum()),
        max_missing_features_per_sample=int(missing.sum(axis=1).max()),
        duplicate_chemical_rows=int(chemistry.duplicated().sum()),duplicate_sensory_rows=int(sensory.duplicated().sum()),
        duplicate_chemical_columns=int(chemistry.T.duplicated().sum()),constant_or_empty_features=int((const<=1).sum()),
        near_identical_sample_pairs=int(near.near_identical.sum()),metadata_id_repairs=len(repairs),
        feature_representation_counts=inventory.representation.value_counts().to_dict(),
        measurement_kind_counts=inventory.measurement_kind.value_counts().to_dict(),
        calibration_cell_status_counts=comparison_frame.status.value_counts().to_dict(),
        invalid_physical_reference_cells=len(physical),
        chemical_statistical_flagged_cells=sum(row['dataset']=='chemical' for row in flagged),
        sensory_statistical_flagged_cells=sum(row['dataset']=='sensory' for row in flagged),
        missingness_style_associations_bh05=int((association.bh_adjusted_p<.05).sum()),
        missingness_style_significant_results=association.loc[association.bh_adjusted_p<.05].to_dict('records'),
        scorecards_audited=len(scorecards),author_workbook_compared=bool(provenance),
        numerical_training_inputs_changed=False,rows_deleted=0,features_deleted=0,
        globally_imputed_cells=0,missing_value_policy='median fitted separately in each training fold',
        outlier_policy='flag and retain; existing training-fold clipping applies to ridge/PLS only',
        scale_policy='keep supplied transforms; no second log, no concentration inversion for uncalibrated signals',
        unavailable_evidence='pre-transform uncalibrated signals, per-compound analytical LOD/LOQ, raw replicate chromatograms',
        verification='passed: source hashes, canonical CSV roundtrip, source alignment, all scorecard defaults')
    (INFO/'cleaning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    cleaning_plot(chemistry,missing,row_flags)
    cleaning_report(summary,chemistry,extreme)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def cleaning_plot(chemistry, missing, row_flags):
    installed={font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':[name for name in ['Microsoft YaHei','SimHei'] if name in installed]+['DejaVu Sans'],
                         'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    counts=chemistry.isna().sum().loc[lambda x:x>0].sort_values()
    axes[0].barh(counts.index,counts.values,color='#0f766e')
    axes[0].set(title='缺失值集中在9个特征',xlabel='缺失测量数')
    categories,frequencies=np.unique(missing.sum(axis=1),return_counts=True)
    axes[1].bar(categories,frequencies,color='#38bdf8')
    axes[1].set(title='每款啤酒的缺失项数',xlabel='缺失特征数',ylabel='样本数',xticks=categories)
    axes[2].hist(row_flags,bins=15,color='#c2410c')
    axes[2].set(title='统计极端值仅标记、保留',xlabel='每样本被标记的化学测量数',ylabel='样本数')
    fig.savefig(FIG/'数据清洗概览.png',dpi=160);plt.close(fig)


def cleaning_report(summary, chemistry, extreme):
    missing=chemistry.isna().sum().loc[lambda x:x>0].sort_values(ascending=False)
    table='\n'.join(f'| {name} | {count} | {100*count/len(chemistry):.2f}% |' for name,count in missing.items())
    status=summary['calibration_cell_status_counts']
    missing_associations='\n'.join(
        f'- {row["feature"]}：{row["style"]}组缺失{row["style_missing"]}/{row["style_total"]}，'
        f'其他风格缺失{row["others_missing"]}/{row["others_total"]}，BH校正p={row["bh_adjusted_p"]:.6f}。'
        for row in summary['missingness_style_significant_results']) or '未检出BH校正后显著的风格关联。'
    extreme_table='\n'.join(
        f'| {row.beer} | {row.feature} | {row.value:.7g} | {row.robust_mad_score:.1f} |'
        for row in extreme.head(5).itertuples())
    source_comparison = ('作者公开S1、S4与比赛1.csv、4.csv的数值及缺失位置匹配；8.xlsx的校准表也与作者原表匹配。'
                         '由此可确认上述跨表差异已经存在于上游公开资料，而非本次CSV读取错误。'
                         if summary['author_workbook_compared'] else
                         '本次本地未提供作者公开工作簿，因此本次只执行附件内部核验，不声明已独立逐值核对作者S1/S4。')
    text=f'''# 第一问数据清洗报告

已完成现有附件可执行的数据清洗、来源追溯及逐项质量核验。保留250个样本、231个特征和50个感官目标；不删除真实但极端的啤酒样本。

## 数据与清洗产物

清洗后的特征表和标签表位于 `数据表格/数据清洗/clean_features.csv`、`clean_targets.csv`。两表按beer对齐并使用化学表和8.xlsx共同确认的精确商品编号，修复了感官表{summary['metadata_id_repairs']}条舍入编号。数值训练输入与原附件保持一致，已核对CSV导出回读和原附件哈希。

缺失值保留为NA；中位数填补、线性/PLS分支的1%—99%裁剪和标准化均在训练折内拟合。清洗表不包含用全体250个样本计算后预先填补的值，避免交叉验证泄露。

## 重复、常量和缺失

- 重复化学特征行：{summary['duplicate_chemical_rows']}；重复感官行：{summary['duplicate_sensory_rows']}；重复化学列：{summary['duplicate_chemical_columns']}。
- 常量或全空特征：{summary['constant_or_empty_features']}；近乎相同的样本对：{summary['near_identical_sample_pairs']}（至少90%特征可比且最大IQR缩放差≤1e-6）。不因风格相同或感官值接近而去重。
- 缺失{summary['missing_cells']}处，占{100*summary['missing_fraction']:.4f}%，影响{summary['affected_samples']}款，每款最多缺{summary['max_missing_features_per_sample']}项。

| 特征 | 缺失数 | 样本占比 |
| --- | --- | --- |
{table}

已输出样本、特征、风格分布、共现模式和缺失掩码。9个缺失特征×22种风格共198个Fisher精确检验经BH多重校正后，显著关联数为{summary['missingness_style_associations_bh05']}；未检出关联不能证明完全随机缺失，不将NA填为0。

{missing_associations}

## 231列的尺度与来源

`feature_scales.csv`逐列列出名称、测量类型、源单位、变换证据和允许的解释。70列有校准参考：37列的正值与log10参考一致；33列以原值表示为主，但部分单元格存在跨表差异，不能仅凭中位数误差宣称全表一致。

另外161列分为157个未校准相对信号和4个合计信号。已确认它们不是可直接当作mg/L的绝对浓度：作者方法说明色谱峰面积经内标归一化，绘图脚本将未校准字段标为A.U.。公开表未提供其变换前逐列信号和完整变换清单，因此保留给定表示，不再次取对数，不反推或虚构浓度。信号类别已判明，精确逆变换不具备数据依据。

## 校准工作表与主表不一致的处置

逐单元格比较使用atol=rtol=1e-7，允许CSV舍入。发现：

- 与参考转换不一致：{status.get('reference_disagreement',0)}处。
- 参考缺失、主表有数值：{status.get('reference_missing_supplied_present',0)}处。
- 参考为0、主表为NA：{status.get('source_zero_to_missing',0)}处，保留主表NA，不将其冒充已确认的真实0。
- 参考≤0、主表对应半个最小正值后取log10：{status.get('nonpositive_reference_half_min_floor',0)}处。这是已有处理痕迹，不能把替代值说成仪器检出限。

{source_comparison} 保留比赛主表/S1，所有差异导出`calibration_cell_audit.csv`，未用辅助表覆盖主表。

## 异常值与物理范围

化学数据中{summary['chemical_statistical_flagged_cells']}个单元格被1.5IQR或6×1.4826MAD规则标记；感官表标记{summary['sensory_statistical_flagged_cells']}个。IQR/MAD为零时不使用该规则。统计标记用于复核，不作为测量错误判据，不自动截断或删行。感官标签是标准化值，负数不等于错误。

最需复核的化学极端值示例（完整前20项及来源状态见`extreme_value_review.csv`）：

| beer | 特征 | 附件数值 | 稳健MAD偏离分数 |
| --- | --- | --- | --- |
{extreme_table}

这些极端值也存在于作者公开处理表。来源一致只能排除本次转录/读取错误，不能证明峰识别、校准或批次处理正确。无原始色谱图和重复测量时，无法确定应替换成哪个真值；保留并标记，线性模型仍采用训练折内裁剪控制外推。

对校准原值检查非负性、pH的0—14范围、乙醇的0—100%范围，发现辅助校准表有{summary['invalid_physical_reference_cells']}个负测量值（β-葡聚糖2处、不可发酵物1处）；对应主表已使用可核对的低端替代值。保留主表，逐项列于`physical_reference_flags.csv`。不使用“常见啤酒范围”删除特殊风格，不把感官阈值误当检测限或异常上界。

全12张评分卡都已检查：公式缓存0为未填写模板的默认结果，不用于感官训练，也不补造没有独立标签的乙醛子项目。

## 结论与适用范围

本轮完成了模型输入所需的清洗与核验，数值训练输入没有变化，现有模型及外层验证指标仍对应同一组输入；本轮未重新训练模型。准确复核测量误差、重做批次校正或反推全部浓度仍需要未公开的原始重复测量/色谱图、逐化合物LOD/LOQ和变换前信号，不能由本轮清洗凭空补齐。

依据：[作者论文方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC10966102/)、[固定版本公开数据与脚本](https://zenodo.org/records/10653704)。来源比对与哈希见`数据表格/数据清洗/published_source_comparison.csv`，处理汇总见`元数据/cleaning_summary.json`。

![清洗概览](绘图/数据清洗概览.png)
'''
    (OUT/'数据清洗报告.md').write_text(text,encoding='utf-8')


class QuantileClipper(BaseEstimator):
    """Bound linear extrapolation using training-only marginal quantiles."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_ = np.quantile(X, self.lower, axis=0)
        self.upper_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


def candidates(seed=20261024):
    models = {}
    for alpha in [10., 100., 1000.]:
        models[f'ridge_{int(alpha)}'] = make_pipeline(SimpleImputer(strategy='median'), QuantileClipper(), StandardScaler(), Ridge(alpha=alpha))
    for n in [5, 12]:
        models[f'pls_{n}'] = make_pipeline(SimpleImputer(strategy='median'), QuantileClipper(), StandardScaler(), PLSRegression(n_components=n, scale=False, max_iter=1000))
    for leaf in [2, 6]:
        tree = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=leaf, max_features=0.7, random_state=seed, n_jobs=2)
        models[f'extra_trees_leaf{leaf}'] = make_pipeline(SimpleImputer(strategy='median'), tree)
    models['hist_gradient_boost'] = make_pipeline(
        SimpleImputer(strategy='median'),
        MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7,
            max_bins=32, learning_rate=0.06, min_samples_leaf=12, l2_regularization=5,
            early_stopping=False, random_state=seed), n_jobs=2))
    models['independent_extra_trees'] = make_pipeline(SimpleImputer(strategy='median'),
        MultiOutputRegressor(ExtraTreesRegressor(n_estimators=120, min_samples_leaf=3,
            max_features=0.7, random_state=seed, n_jobs=1), n_jobs=2))
    return {name: TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
            for name, model in models.items()}


class SelectedSensoryModel(BaseEstimator, RegressorMixin):
    """One selected candidate per target, fitted using only the supplied training set."""
    def __init__(self, seed=20261024, inner_splits=3):
        self.seed = seed
        self.inner_splits = inner_splits

    def fit(self, X, Y):
        self.names_ = list(candidates(self.seed))
        self.cv_predictions_ = np.zeros((len(self.names_), *Y.shape))
        cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=self.seed)
        for tr, va in cv.split(X):
            for j, (name, estimator) in enumerate(candidates(self.seed).items()):
                estimator.fit(X[tr], Y[tr])
                self.cv_predictions_[j, va] = estimator.predict(X[va])
        for blend_name, components in [
                ('blend_tree_linear', ['ridge_100', 'extra_trees_leaf2', 'hist_gradient_boost']),
                ('blend_tree_only', ['independent_extra_trees', 'extra_trees_leaf2', 'hist_gradient_boost'])]:
            blend_ix = [self.names_.index(n) for n in components]
            self.cv_predictions_ = np.concatenate([self.cv_predictions_, self.cv_predictions_[blend_ix].mean(axis=0)[None]], axis=0)
            self.names_.append(blend_name)
        errors = np.mean((self.cv_predictions_ - Y[None, :, :]) ** 2, axis=1)
        # A train-fold mean is a valid candidate for weakly predictable targets.
        mean_preds = np.zeros_like(Y)
        for tr, va in cv.split(X):
            mean_preds[va] = Y[tr].mean(axis=0)
        self.cv_predictions_ = np.concatenate([self.cv_predictions_, mean_preds[None]], axis=0)
        self.names_.append('training_mean')
        errors = np.concatenate([errors, np.mean((mean_preds - Y)**2, axis=0)[None]], axis=0)
        self.selected_ = errors.argmin(axis=0)
        self.inner_mse_ = errors
        self.mean_ = Y.mean(axis=0)
        self.y_reference_ = np.sort(Y, axis=0)
        self.y_sd_ = Y.std(axis=0, ddof=1)
        self.estimators_ = {}
        for name, estimator in candidates(self.seed).items():
            # All are fitted for clean family comparisons; no outer test target is used.
            estimator.fit(X, Y)
            self.estimators_[name] = estimator
        self.n_features_in_ = X.shape[1]
        return self

    def all_predictions(self, X):
        predictions = [est.predict(X) for est in self.estimators_.values()]
        base_names = list(self.estimators_)
        for components in [['ridge_100', 'extra_trees_leaf2', 'hist_gradient_boost'],
                           ['independent_extra_trees', 'extra_trees_leaf2', 'hist_gradient_boost']]:
            predictions.append(np.mean([predictions[base_names.index(n)] for n in components], axis=0))
        predictions.append(np.tile(self.mean_, (len(X), 1)))
        return np.stack(predictions)

    def predict(self, X):
        all_p = self.all_predictions(X)
        return all_p[self.selected_, :, np.arange(len(self.selected_))].T

    def grades(self, prediction):
        return relative_grades(prediction, self.y_reference_)


def relative_grades(values, reference):
    """Seven within-attribute quantile bins; grade 0 DOES NOT mean absence."""
    values = np.asarray(values)
    result = np.zeros_like(values, dtype=np.int64)
    for j in range(values.shape[1]):
        ref = np.sort(reference[:, j])
        # Midrank convention handles ties without assigning every minimum to grade 0.
        q = (np.searchsorted(ref, values[:, j], side='left') +
             np.searchsorted(ref, values[:, j], side='right')) / (2 * len(ref))
        result[:, j] = np.clip(np.floor(7 * q), 0, 6).astype(int)
    return result


def z_anchor_grades(values, mean, sd):
    """Sensitivity alternative: -2 training SD -> 0, mean -> 3, +2 SD -> 6."""
    return np.clip(np.rint(3 + 1.5 * (values - mean) / np.maximum(sd, 1e-12)), 0, 6).astype(int)


class SevenPointIntensityScale:
    def __init__(self, upper_quantile=0.95, tolerance=1e-8):
        self.upper_quantile = upper_quantile
        self.tolerance = tolerance

    def fit(self, Y, targets, beer_ids=None):
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or not np.isfinite(Y).all() or Y.shape[1] != len(targets):
            raise ValueError('Finite 2-D targets with matching names are required')
        if not 0.5 < self.upper_quantile <= 1:
            raise ValueError('upper_quantile must be above 0.5 and at most 1')
        self.targets_ = list(targets)
        beer_ids = list(map(str, range(len(Y)))) if beer_ids is None else list(map(str, beer_ids))
        if len(beer_ids) != len(Y):
            raise ValueError('Beer IDs do not match training rows')
        self.anchors_ = np.zeros((Y.shape[1], 3), dtype=float)
        self.anchor_rows_ = []
        self.status_ = 'unverified_empirical_reference_candidates'
        for j, name in enumerate(targets):
            values = Y[:, j]
            lo = float(values.min())
            eligible = np.flatnonzero(values > lo + self.tolerance)
            floor_count = int(np.sum(np.abs(values-lo) <= self.tolerance))
            low_ix = int(np.argmin(values))
            if len(eligible) < 2:
                if len(eligible):
                    hi_ix = int(eligible[0]); hi = float(values[hi_ix])
                    mid = (lo+hi)/2; mid_ix = None
                    mode = 'two_anchor_fallback'
                else:
                    hi = lo; mid = lo; hi_ix = low_ix; mid_ix = None
                    mode = 'no_identifiable_intensity_range'
            else:
                mid_q = float(np.median(values[eligible]))
                hi_q = float(np.quantile(values[eligible], self.upper_quantile))
                mid_ix = int(eligible[np.argmin(np.abs(values[eligible]-mid_q))])
                hi_ix = int(eligible[np.argmin(np.abs(values[eligible]-hi_q))])
                mid, hi = float(values[mid_ix]), float(values[hi_ix])
                mode = 'three_reference_candidates'
                if hi <= mid + self.tolerance:
                    hi_ix = int(np.argmax(values)); hi = float(values[hi_ix])
                if hi <= mid + self.tolerance:
                    mid = (lo+hi)/2; mid_ix = None; mode = 'two_anchor_fallback'
            self.anchors_[j] = [lo, mid, hi]
            for anchor, number, val, ix in [('low', 0, lo, low_ix), ('middle', 3, mid, mid_ix), ('high', 6, hi, hi_ix)]:
                self.anchor_rows_.append({
                    'target': name, 'anchor': anchor, 'assumed_intensity_score': number,
                    'provided_scale_value': val, 'reference_candidate_beer': beer_ids[ix] if ix is not None else '',
                    'anchor_status': self.status_, 'selection_mode': mode,
                    'training_floor_count': floor_count, 'training_rows': len(Y),
                    'upper_quantile_among_above_floor': self.upper_quantile,
                    'sensory_calibration_confirmed': False,
                })
        return self

    @classmethod
    def from_anchor_table(cls, table, targets):
        """Build a scale from externally established anchors; validates order.

        Required fields: target, anchor (low/middle/high), provided_scale_value.
        Only a table explicitly marked confirmed may be described as confirmed.
        """
        obj = cls()
        obj.targets_ = list(targets)
        values=[]
        for name in targets:
            part=table[table.target==name].set_index('anchor')
            if not part.index.is_unique or not {'low','middle','high'} <= set(part.index):
                raise ValueError(f'Missing or repeated anchor for {name}')
            row=part.loc[['low','middle','high'],'provided_scale_value'].to_numpy(float)
            if not np.isfinite(row).all() or not (row[0] < row[1] < row[2]):
                raise ValueError(f'Anchor values must increase for {name}')
            values.append(row)
        obj.anchors_=np.asarray(values)
        obj.anchor_rows_=table.to_dict('records')
        obj.status_='external_reference_table'
        return obj

    def scores(self, prediction):
        prediction = np.asarray(prediction, dtype=float)
        if prediction.ndim != 2 or prediction.shape[1] != len(self.targets_) or not np.isfinite(prediction).all():
            raise ValueError('Prediction shape or finite-value check failed')
        result=np.zeros_like(prediction)
        for j,(lo,mid,hi) in enumerate(self.anchors_):
            if hi <= lo + self.tolerance:
                result[:,j]=np.nan
            else:
                result[:,j]=np.interp(prediction[:,j],[lo,mid,hi],[0.,3.,6.],left=0.,right=6.)
        return result

    def grades(self, prediction):
        scores=self.scores(prediction)
        if not np.isfinite(scores).all():
            raise ValueError('At least one target lacks an identifiable intensity range')
        return np.floor(scores+0.5).astype(np.int64)


# ---------- Training, evaluation and output ----------

def csv(frame, name):
    frame.to_csv(TABLES / name, index=False, encoding='utf-8-sig')

def score(y, p):
    return {'macro_r2': float(r2_score(y, p, multioutput='uniform_average')),
            'mae_provided_scale': float(np.abs(y-p).mean()),
            'rmse_provided_scale': float(np.sqrt(np.mean((y-p)**2)))}

def train():
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    INFO.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    SCORE_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    meta, X, Y, features, targets, mapping, audit = load_data(TABLES)
    n, q = Y.shape
    core = np.array([t.startswith(('A_', 'F_')) for t in targets])
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    test_count = np.zeros(n, dtype=int)
    for tr, te in folds:
        assert not set(tr) & set(te)
        test_count[te] += 1
    assert (test_count == 1).all()
    (INFO / 'protocol.json').write_text(json.dumps({
        'seed': SEED, 'outer_splits': 5, 'inner_splits': 3,
        'target_count': 50, 'primary_core_target_count': 30,
        'model_selection': 'minimum inner-fold OOF MSE independently per target',
        'excluded_predictors': ['beer', 'beer_id', 'tasting_category_fine', '4.csv', '7.csv', 'external model outputs'],
        'missingness_scenarios': [0.0, 0.1, 0.2, 0.3], 'mask_repeats': 5,
        'stress_test': 'additional MCAR masking of observed outer-test feature cells; no train/test refit',
        'grade_mapping': 'QDP-inspired seven-point proxy anchors; quantile bins exported only as historical comparison',
        'folds': [{'fold': i+1, 'train_beer': meta.beer.iloc[tr].tolist(),
                   'test_beer': meta.beer.iloc[te].tolist()} for i, (tr,te) in enumerate(folds)],
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                        'numpy': np.__version__, 'pandas': pd.__version__}}, ensure_ascii=False, indent=2), encoding='utf-8')

    oof = np.zeros_like(Y)
    grades = np.zeros_like(Y, dtype=int)
    zgrades = np.zeros_like(Y, dtype=int)
    fold_labels = np.zeros(n, dtype=int)
    family_oof = {}
    choices = []
    stress_rows = []
    fold_rows = []
    for fold, (tr, te) in enumerate(folds, 1):
        print(f'Fold {fold}/5: train={len(tr)}, test={len(te)}, selecting on training only...', flush=True)
        model = SelectedSensoryModel(seed=SEED+fold).fit(X[tr], Y[tr])
        all_p = model.all_predictions(X[te])
        prediction = all_p[model.selected_, :, np.arange(q)].T
        grade = model.grades(prediction)
        zgrade = z_anchor_grades(prediction, model.mean_, model.y_sd_)

        # A separately specified robustness comparator: duplicate only training
        # samples, mask their features, keep their labels. Never split duplicates.
        rng = np.random.default_rng(SEED + fold*100)
        aug_x = [X[tr]]
        for rate in [0.1, 0.2, 0.3]:
            masked = X[tr].copy(); masked[rng.random(masked.shape) < rate] = np.nan
            aug_x.append(masked)
        robust = TransformedTargetRegressor(regressor=make_pipeline(
            SimpleImputer(strategy='median'), ExtraTreesRegressor(n_estimators=300,
                min_samples_leaf=6, max_features=0.7, random_state=SEED+fold, n_jobs=2)),
            transformer=StandardScaler()).fit(np.vstack(aug_x), np.tile(Y[tr], (4,1)))
        stress = []
        for rate in [0.0, 0.1, 0.2, 0.3]:
            repeats = 1 if rate == 0 else 5
            for repeat in range(repeats):
                rng = np.random.default_rng(SEED + fold*1000 + repeat)
                masked = X[te].copy()
                masked[rng.random(masked.shape) < rate] = np.nan
                for method, p in [('nested_selected', model.predict(masked)),
                                  ('mask_augmented_extra_trees', robust.predict(masked))]:
                    stress.append(dict(fold=fold, additional_missing_rate=rate, repeat=repeat,
                                       method=method, **score(Y[te],p), core_macro_r2=score(Y[te][:,core],p[:,core])['macro_r2']))
        rec = dict(te=te, prediction=prediction, grades=grade, zgrades=zgrade,
                   families={name:all_p[j] for j,name in enumerate(model.names_)},
                   choices=[model.names_[j] for j in model.selected_], stress=stress)
        oof[te] = rec['prediction']; grades[te] = rec['grades']; zgrades[te] = rec['zgrades']
        fold_labels[te] = fold
        for name, pred in rec['families'].items():
            family_oof.setdefault(name, np.zeros_like(Y))[te] = pred
        choices.extend(dict(fold=fold,target=t,selected_model=name) for t,name in zip(targets,rec['choices']))
        stress_rows.extend(rec['stress'])
        fold_rows.append(dict(fold=fold, **score(Y[te],oof[te]), core_macro_r2=score(Y[te][:,core],oof[te][:,core])['macro_r2']))
        print('Fold result:',fold_rows[-1],f'elapsed={time.time()-started:.1f}s',flush=True)

    result = score(Y, oof)
    result['core_macro_r2'] = score(Y[:,core],oof[:,core])['macro_r2']
    baseline = family_oof['training_mean']
    result['rmse_reduction_vs_training_mean'] = 1 - result['rmse_provided_scale']/score(Y,baseline)['rmse_provided_scale']
    result['relative_grade_mapping_disagreement_fraction'] = float((grades != zgrades).mean())
    result['grade_mae_between_mapping_assumptions'] = float(np.abs(grades-zgrades).mean())
    family_rows = [dict(model=name,**score(Y,p),core_macro_r2=score(Y[:,core],p[:,core])['macro_r2'])
                   for name,p in family_oof.items()]
    family_rows.append(dict(model='nested_selected',**score(Y,oof),core_macro_r2=result['core_macro_r2']))
    csv(pd.DataFrame(family_rows), 'model_comparison.csv')
    csv(pd.DataFrame(fold_rows), 'outer_fold_metrics.csv')
    csv(pd.DataFrame(choices), 'outer_model_selection.csv')
    stress_df = pd.DataFrame(stress_rows); csv(stress_df,'missingness_stress_raw.csv')
    # Average repeats within a fold first, so repeats are not treated as independent datasets.
    stress_fold = stress_df.groupby(['method','additional_missing_rate','fold'],as_index=False)[
        ['macro_r2','core_macro_r2','rmse_provided_scale','mae_provided_scale']].mean()
    stress_summary=stress_fold.groupby(['method','additional_missing_rate']).agg(
        macro_r2=('macro_r2','mean'), core_macro_r2=('core_macro_r2','mean'),
        rmse=('rmse_provided_scale','mean'), mae=('mae_provided_scale','mean'),
        fold_sd_r2=('macro_r2','std')).reset_index()
    csv(stress_summary,'missingness_stress_summary.csv')
    target_rows=[]
    for j,t in enumerate(targets):
        target_rows.append(dict(target=t,scope='core_aroma_flavour' if core[j] else 'additional',
            r2=float(r2_score(Y[:,j],oof[:,j])),mae=float(np.abs(Y[:,j]-oof[:,j]).mean()),
            rmse=float(np.sqrt(np.mean((Y[:,j]-oof[:,j])**2))),
            normalized_rmse=float(np.sqrt(np.mean((Y[:,j]-oof[:,j])**2))/Y[:,j].std(ddof=1)),
            spearman=float(spearmanr(Y[:,j],oof[:,j]).statistic),
            baseline_r2=float(r2_score(Y[:,j],baseline[:,j]))))
    csv(pd.DataFrame(target_rows).merge(mapping,on=['target','scope']), 'per_target_metrics.csv')
    csv(pd.concat([meta,pd.Series(fold_labels,name='outer_fold'),pd.DataFrame(oof,columns=targets)],axis=1),'oof_predictions_continuous.csv')
    csv(pd.concat([meta,pd.Series(fold_labels,name='outer_fold'),pd.DataFrame(grades,columns=targets)],axis=1),'oof_relative_grades_0_6.csv')
    csv(pd.concat([meta,pd.DataFrame(zgrades,columns=targets)],axis=1),'oof_alternative_grades_0_6.csv')
    csv(pd.concat([meta,pd.DataFrame(Y,columns=targets)],axis=1),'reference_targets_provided_scale.csv')
    np.savez_compressed(INFO/'evaluation_arrays.npz',y=Y,prediction=oof,baseline=baseline,grades=grades,fold=fold_labels)

    # Sample bootstrap summarizes empirical uncertainty of the completed nested OOF
    # predictions. This does not include variability from retraining the whole procedure.
    rng=np.random.default_rng(SEED); boots=[]
    for _ in range(1000):
        ix=rng.integers(0,n,n)
        boots.append([r2_score(Y[ix],oof[ix]),r2_score(Y[ix][:,core],oof[ix][:,core])])
    ci=np.quantile(boots,[0.025,0.975],axis=0)
    result['bootstrap_macro_r2_95_interval_fixed_oof'] = ci[:,0].tolist()
    result['bootstrap_core_r2_95_interval_fixed_oof'] = ci[:,1].tolist()

    print('Running separate split-conformal experiment (150 fit / 50 calibration / 50 test)...',flush=True)
    all_ix=np.arange(n)
    fitcal,test=train_test_split(all_ix,test_size=50,random_state=SEED+999)
    fit,cal=train_test_split(fitcal,test_size=50,random_state=SEED+1000)
    assert not set(fit)&set(cal) and not set(fit)&set(test) and not set(cal)&set(test)
    conformal=SelectedSensoryModel(seed=SEED+2000).fit(X[fit],Y[fit])
    alpha=0.10
    residual=np.abs(Y[cal]-conformal.predict(X[cal]))
    kth=min(len(cal),int(np.ceil((len(cal)+1)*(1-alpha))))
    radius=np.sort(residual,axis=0)[kth-1]
    point=conformal.predict(X[test]); lower=point-radius; upper=point+radius
    coverage=((Y[test]>=lower)&(Y[test]<=upper)).mean(axis=0)
    csv(pd.DataFrame({'target':targets,'nominal_coverage':1-alpha,'test_coverage':coverage,
                      'interval_width':2*radius}), 'split_conformal_metrics.csv')
    records=[]
    for i,ix in enumerate(test):
        for j,t in enumerate(targets):
            records.append(dict(beer=meta.beer.iloc[ix],target=t,observed=Y[ix,j],prediction=point[i,j],
                                lower=lower[i,j],upper=upper[i,j]))
    csv(pd.DataFrame(records),'split_conformal_test_predictions.csv')
    result['split_conformal_90_mean_coverage']=float(coverage.mean())
    result['split_conformal_90_core_mean_coverage']=float(coverage[core].mean())
    joblib.dump(dict(model=conformal,radius=radius,alpha=alpha,features=features,targets=targets,
                     fit_beers=meta.beer.iloc[fit].tolist(),calibration_beers=meta.beer.iloc[cal].tolist(),
                     test_beers=meta.beer.iloc[test].tolist()),MODEL_DIR/'split_conformal.joblib')

    print('Fitting reusable model on all 250 samples; these fitted values are NOT evaluation results...',flush=True)
    final=SelectedSensoryModel(seed=SEED).fit(X,Y)
    joblib.dump(dict(model=final,features=features,targets=targets,seed=SEED,
                     grade_notice='provisional relative quantile bins; not recovered original 0-6 scores',
                     feature_notice='use exactly the input CSV representation; no second log transform'),
                MODEL_DIR/'sensory_model.joblib')
    csv(pd.DataFrame({'target':targets,'selected_model':[final.names_[j] for j in final.selected_]}),'final_model_selection.csv')
    csv(pd.DataFrame(final.inner_mse_.T,columns=final.names_,index=targets).reset_index(names='target'),'final_inner_cv_mse.csv')
    csv(pd.DataFrame(final.y_reference_,columns=targets),'grade_reference_distribution.csv')
    # Serialization, input/output alignment, and finite predictions are material checks.
    restored=joblib.load(MODEL_DIR/'sensory_model.joblib')
    np.testing.assert_allclose(final.predict(X[:7]),restored['model'].predict(X[:7]),rtol=0,atol=1e-12)
    assert np.isfinite(oof).all() and np.isfinite(final.predict(X)).all()
    assert grades.min()>=0 and grades.max()<=6
    for entry in audit['input_manifest']:
        import hashlib
        assert hashlib.sha256((attachment_dir()/entry['file']).read_bytes()).hexdigest()==entry['sha256']
    result['elapsed_seconds']=round(time.time()-started,2)
    result['verification']='passed: nested fold isolation, finite predictions, grade bounds, serialization, unchanged source hashes'
    (INFO/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

# ---------- Seven-point intensity mapping and sensitivity ----------

def save(frame,name):
    frame.to_csv(SCORE_OUT/name,index=False,encoding='utf-8-sig')

def calibrate():
    SCORE_OUT.mkdir(parents=True,exist_ok=True)
    meta,X,Y,features,all_targets,mapping,audit=load_data()
    base=json.loads((INFO/'protocol.json').read_text(encoding='utf-8'))
    arrays=np.load(INFO/'evaluation_arrays.npz')
    np.testing.assert_allclose(arrays['y'],Y,rtol=0,atol=0)
    # Overall appreciation is not a perceptual-intensity attribute.
    keep=np.array([t!='overall' for t in all_targets])
    targets=[t for t,k in zip(all_targets,keep) if k]
    core=np.array([t.startswith(('A_','F_')) for t in targets])
    Y=Y[:,keep];P=arrays['prediction'][:,keep]
    assert Y.shape==(250,49) and core.sum()==30
    scores=np.zeros_like(Y);grades=np.zeros_like(Y,dtype=int)
    bootstrap_low=np.zeros_like(Y);bootstrap_high=np.zeros_like(Y)
    alternatives={q:np.zeros_like(Y,dtype=int) for q in [.90,.99]}
    references=[]
    fold_ids=np.zeros(len(Y),dtype=int)
    n_boot=150
    for fold in base['folds']:
        tr=np.flatnonzero(meta.beer.isin(fold['train_beer']).to_numpy())
        te=np.flatnonzero(meta.beer.isin(fold['test_beer']).to_numpy())
        assert not set(tr)&set(te)
        scale=SevenPointIntensityScale().fit(Y[tr],targets,meta.beer.iloc[tr])
        assert np.isfinite(scale.scores(P[te])).all()
        scores[te]=scale.scores(P[te]);grades[te]=scale.grades(P[te]);fold_ids[te]=fold['fold']
        for row in scale.anchor_rows_:
            row.update(fit_scope=f'outer_train_fold_{fold["fold"]}')
            if row['reference_candidate_beer']:
                assert row['reference_candidate_beer'] in fold['train_beer']
                assert row['reference_candidate_beer'] not in fold['test_beer']
        references.extend(scale.anchor_rows_)
        for q in alternatives:
            alt=SevenPointIntensityScale(upper_quantile=q).fit(Y[tr],targets,meta.beer.iloc[tr])
            alternatives[q][te]=alt.grades(P[te])
        rng=np.random.default_rng(20261025+fold['fold'])
        boot=[]
        for _ in range(n_boot):
            ix=rng.choice(tr,size=len(tr),replace=True)
            b=SevenPointIntensityScale().fit(Y[ix],targets)
            # Constant bootstrap attributes get the main mapping instead of fake zeros.
            s=b.scores(P[te]);invalid=~np.isfinite(s)
            s[invalid]=scores[te][invalid]
            boot.append(np.floor(s+.5).astype(int))
        bootstrap_low[te],bootstrap_high[te]=np.quantile(boot,[.025,.975],axis=0,method='nearest')
        print('Scoring fold',fold['fold'],'complete',flush=True)
    assert np.isfinite(scores).all() and scores.min()>=0 and scores.max()<=6
    assert grades.min()>=0 and grades.max()<=6
    prefix=pd.concat([meta,pd.Series(fold_ids,name='outer_fold')],axis=1)
    for name,values,columns in [
        ('oof_estimated_intensity_scores_0_6.csv',scores,targets),
        ('oof_estimated_intensity_grades_0_6.csv',grades,targets),
        ('oof_core_aroma_flavour_grades_0_6.csv',grades[:,core],np.array(targets)[core]),
        ('anchor_bootstrap_grade_lower.csv',bootstrap_low.astype(int),targets),
        ('anchor_bootstrap_grade_upper.csv',bootstrap_high.astype(int),targets)]:
        save(pd.concat([prefix,pd.DataFrame(values,columns=columns)],axis=1),name)
    card=prefix[prefix.beer.astype(int).between(1,12)].copy()
    card_grades=pd.DataFrame(grades,columns=targets).loc[card.index]
    card=pd.concat([card,card_grades],axis=1)
    card['order']=card.beer.astype(int)
    save(card.sort_values('order').drop(columns='order'),'scorecard_beer_1_to_12_estimated_grades.csv')
    metrics=pd.read_csv(TABLES/'per_target_metrics.csv',header=0).set_index('target')
    sensitivity=[]
    for j,name in enumerate(targets):
        sensitivity.append({'target':name,'scope':'core_aroma_flavour' if core[j] else 'additional_intensity',
            'predictive_oof_r2_provided_scale':metrics.loc[name,'r2'],
            'grade_change_fraction_upper_q90':float(np.mean(grades[:,j]!=alternatives[.90][:,j])),
            'grade_change_fraction_upper_q99':float(np.mean(grades[:,j]!=alternatives[.99][:,j])),
            'mean_anchor_bootstrap_grade_range':float(np.mean(bootstrap_high[:,j]-bootstrap_low[:,j]))})
    save(pd.DataFrame(sensitivity),'per_target_mapping_sensitivity.csv')
    full=SevenPointIntensityScale().fit(Y,targets,meta.beer)
    for row in full.anchor_rows_:row['fit_scope']='all_250_for_future_predictions'
    references.extend(full.anchor_rows_)
    refs=pd.DataFrame(references)
    save(refs,'reference_candidates_and_assumptions.csv')
    dictionary=mapping[mapping.target!='overall'].copy()
    dictionary['scale']='QDP-inspired seven-point ordinal intensity'
    dictionary['calibration_status']='unverified_empirical_reference_candidates'
    save(dictionary,'intensity_target_dictionary.csv')
    semantic=pd.DataFrame({'grade':list(range(7)),
        'adopted_project_label':['未感知','极弱','弱','中等','较强','强','极强／占主导'],
        'source_status':['题目评分卡端点']+['项目中间等级用语']*5+['题目评分卡端点'],
        'note':['模型估计依赖参照假设；不等于实验证明该风味不存在']+['非国标逐级原文定义']*5+['高端候选参照未接受人工强度校准']})
    save(semantic,'adopted_seven_point_scale.csv')
    joblib.dump({'scale':full,'targets':targets,'source_targets':all_targets,'keep_mask':keep.tolist(),
        'method':'QDP + seven-point intensity format used by source research',
        'calibration_status':full.status_,
        'notice':'Estimated scores under empirical proxy anchors; not original human scores or certified standard-calibrated measurements'},
        MODEL_DIR/'seven_point_scale.joblib')
    restored=joblib.load(MODEL_DIR/'seven_point_scale.joblib')
    np.testing.assert_array_equal(full.grades(P[:7]),restored['scale'].grades(P[:7]))
    # Check monotonicity and actual endpoints for each target.
    for j,(lo,mid,hi) in enumerate(full.anchors_):
        sample=np.tile(full.anchors_[:,1],(101,1));sample[:,j]=np.linspace(lo-1,hi+1,101)
        g=full.grades(sample)[:,j]
        assert np.all(np.diff(g)>=0) and g[0]==0 and g[-1]==6
        assert set(refs[(refs.target==targets[j])&(refs.fit_scope=='all_250_for_future_predictions')].assumed_intensity_score)=={0,3,6}
    summary={'borrowed_format':'source-study seven-point intensity / QDP',
        'samples':250,'core_aroma_flavour_targets':30,'all_intensity_targets':49,
        'excluded':'overall appreciation; source scorecard acetaldehyde subfields have no matching direct training labels',
        'absolute_sensory_calibration_confirmed':False,
        'proxy_anchor_rule':'training minimum -> assumed 0; median above floor -> assumed 3; actual observation nearest 95th percentile above floor -> assumed 6',
        'middle_grade_labels':'project wording, not quoted standard definitions',
        'grade_change_fraction_q90':float(np.mean(grades!=alternatives[.90])),
        'grade_change_fraction_q99':float(np.mean(grades!=alternatives[.99])),
        'anchor_bootstrap_repeats_per_fold':n_boot,
        'fraction_cells_with_nonzero_anchor_bootstrap_range':float(np.mean(bootstrap_high>bootstrap_low)),
        'mean_anchor_bootstrap_grade_range':float(np.mean(bootstrap_high-bootstrap_low)),
        'grade_frequency':{str(k):int(np.sum(grades==k)) for k in range(7)},
        'original_grade_accuracy':'not measurable without original labels; no proxy accuracy is claimed',
        'continuous_model_retrained':False,
        'continuous_oof_macro_r2':float(json.loads((INFO/'summary.json').read_text(encoding='utf-8'))['macro_r2']),
        'verification':'passed: outer-training anchor isolation, ordinal monotonicity, endpoint/bounds and serialization checks'}
    (INFO/'seven_point_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

# ---------- Figures ----------

def plot_results():
    summary=json.loads((INFO/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((INFO/'audit.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(TABLES/'per_target_metrics.csv',header=0)
    compare=pd.read_csv(TABLES/'model_comparison.csv',header=0)
    stress=pd.read_csv(TABLES/'missingness_stress_summary.csv',header=0)
    interval=pd.read_csv(TABLES/'split_conformal_metrics.csv',header=0)
    target_names=metrics.set_index('target').CleanName_PanelFeedback.to_dict()
    arrays=np.load(INFO/'evaluation_arrays.npz')
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


def write_report():
    """Write the current results without depending on old reports or downloads."""
    summary = json.loads((INFO/'summary.json').read_text(encoding='utf-8'))
    scoring = json.loads((INFO/'seven_point_summary.json').read_text(encoding='utf-8'))
    origin = ('本目录复用此前已完成的完整实验结果和模型，用于单文件整合核对；'
              '整合版独立重训了第一折，四项指标与旧版一致。当前并未重新训练全部五折。'
              if summary.get('result_origin') == 'previous_complete_experiment'
              else '本次从原始附件重新完成数据核对、建模、评价、七点评分、图表和模型导出。')
    report = f"""# A题第一问：单文件运行结果

{origin}

- 输入：250个样本，231个化学/理化特征；连续目标为50个感官属性。
- 验证：外层5折评价，内层3折逐属性选择模型；插补、裁剪、标准化均在训练折拟合。
- 50项宏平均R²：{summary['macro_r2']:.6f}。
- 30项核心香气/风味宏平均R²：{summary['core_macro_r2']:.6f}。
- 连续值MAE：{summary['mae_provided_scale']:.6f}；RMSE：{summary['rmse_provided_scale']:.6f}。
- 独立区间实验：150个训练、50个校准、50个测试，逐属性90%区间平均覆盖率为{summary['split_conformal_90_mean_coverage']:.4f}。

候选模型包含岭回归、PLS、极端随机树、梯度提升及固定等权融合；各属性仅按内层验证误差选择。
额外缺失实验遮蔽外层测试集10%、20%、30%的测量，另与仅使用训练数据构建的缺失增强模型比较。
R²的自助区间基于固定的外层预测，不包含整套模型重新训练的变异。
区间实验使用独立模型，报告逐属性覆盖率，不代表50项同时覆盖。

## 七点评分

借鉴QDP参照描述方法及来源研究的七点强度形式，等级为0—6。
0、3、6的经验候选锚点依次取训练标签的最小值、超过最低值部分的中位数附近实际观测、95%分位数附近实际观测，分段线性插值后四舍五入。
每个外层测试样本的映射只用该折训练标签拟合。最终全样本映射供后续新样本使用。
排除overall喜好度后输出49项强度，其中30项为核心香气/风味。

这些锚点是未经过人工强度校准的建模假设。附件标签已经标准化，不能据此宣称恢复了原始人工0—6分，也没有可验证的原始等级准确率。
题目评分卡中没有对应直接训练标签的独立乙醛子项目不补造评分。
中间等级中文名称是项目用语，不是标准逐级定义的引文。

- 高端锚点改为90%分位数时，等级改变比例：{scoring['grade_change_fraction_q90']:.4f}。
- 高端锚点改为99%分位数时，等级改变比例：{scoring['grade_change_fraction_q99']:.4f}。
- 锚点自助抽样次数：每折{scoring['anchor_bootstrap_repeats_per_fold']}次。
- 参考研究：[Schreurs等，2024](https://doi.org/10.1038/s41467-024-46346-0)。

## 结果文件

-- `数据表格/七点评分/oof_core_aroma_flavour_grades_0_6.csv`：250×30核心评分表。
- `数据表格/七点评分/oof_estimated_intensity_grades_0_6.csv`：250×49扩展强度评分表。
- `数据表格/七点评分/scorecard_beer_1_to_12_estimated_grades.csv`：题目12款啤酒评分。
- `数据表格/七点评分/reference_candidates_and_assumptions.csv`：候选参照及假设。
- `数据表格/oof_predictions_continuous.csv`：50项连续感官属性的外层预测。
- `数据表格/per_target_metrics.csv`、`model_comparison.csv`：逐属性及模型比较。
- `数据表格/missingness_stress_summary.csv`：缺失压力实验。
- `模型/question1_model.joblib`：主模型、七点映射及独立区间模型的统一文件。
- `元数据/audit.json`：输入核对与原始文件哈希。

相对分位数等级文件仅保留作历史方案对照，当前使用上述七点强度表。

![结果概览](绘图/question1_overview.png)
"""
    (OUT/'结果说明.md').write_text(report, encoding='utf-8')


def combine_models():
    """Create one reusable artifact with no dependency on old source modules."""
    bundle = joblib.load(MODEL_DIR/'sensory_model.joblib')
    bundle['scoring'] = joblib.load(MODEL_DIR/'seven_point_scale.joblib')
    bundle['intervals'] = joblib.load(MODEL_DIR/'split_conformal.joblib')
    bundle['grade_notice'] = bundle['scoring']['notice']
    bundle['format_version'] = 1
    target = MODEL_DIR/'question1_model.joblib'
    joblib.dump(bundle, target, compress=3)
    restored = joblib.load(target)
    _, X, *_ = load_data()
    original = bundle['model'].predict(X[:7])
    actual = restored['model'].predict(X[:7])
    np.testing.assert_allclose(original, actual, rtol=0, atol=1e-12)
    keep = restored['scoring']['keep_mask']
    np.testing.assert_array_equal(
        bundle['scoring']['scale'].grades(original[:,keep]),
        restored['scoring']['scale'].grades(actual[:,keep]))
    # These three intermediate files were created by this run and have been merged.
    for name in ['sensory_model.joblib','seven_point_scale.joblib','split_conformal.joblib']:
        (MODEL_DIR/name).unlink()
    if not any(MODEL_DIR.iterdir()):
        MODEL_DIR.rmdir()


def predict_file(source, model_path, output):
    """Predict new measurements using only this file and its unified model."""
    source, output = source.resolve(), output.resolve()
    bundle = joblib.load(model_path)
    frame = pd.read_csv(source, header=0, dtype={c:str for c in META})
    if frame.empty or not (set(frame.columns) & set(bundle['features'])):
        raise ValueError('输入为空或没有可识别的化学特征列。')
    X = frame.reindex(columns=bundle['features']).apply(pd.to_numeric,errors='raise').to_numpy(float)
    if np.isinf(X).any() or np.isnan(X).all(axis=1).any():
        raise ValueError('每行至少需要一项有效化学测量，且不能包含无穷值。')
    meta = frame[[c for c in META if c in frame]].copy()
    meta['missing_fraction'] = np.isnan(X).mean(axis=1)
    point = bundle['model'].predict(X)
    scale = bundle['scoring']
    intensity = point[:,scale['keep_mask']]
    interval = bundle['intervals']
    interval_point = interval['model'].predict(X)
    exports = [
        ('predictions_continuous.csv',point,bundle['targets']),
        ('estimated_intensity_scores_0_6.csv',scale['scale'].scores(intensity),scale['targets']),
        ('estimated_intensity_grades_0_6.csv',scale['scale'].grades(intensity),scale['targets']),
        ('interval_model_point.csv',interval_point,interval['targets']),
        ('interval_lower_90.csv',interval_point-interval['radius'],interval['targets']),
        ('interval_upper_90.csv',interval_point+interval['radius'],interval['targets'])]
    if any((output/name).resolve()==source for name,_,_ in exports):
        raise ValueError('输出路径会覆盖输入文件。')
    output.mkdir(parents=True,exist_ok=True)
    for name,values,columns in exports:
        if not np.isfinite(values).all():
            raise ValueError(f'预测存在无效值：{name}')
        pd.concat([meta,pd.DataFrame(values,columns=columns)],axis=1).to_csv(
            output/name,index=False,encoding='utf-8-sig')
    notes = {
        'grade_notice':bundle['grade_notice'],
        'input_notice':bundle['feature_notice'],
        'unavailable_feature_columns':sorted(set(bundle['features'])-set(frame)),
        'missing_fraction_above_stress_test_range_rows':np.flatnonzero(np.isnan(X).mean(axis=1)>.30).tolist(),
        'interval_notice':'逐属性90%区间由独立训练/校准模型生成；区间中心见interval_model_point.csv，假设新样本可交换且无分布变化。'}
    (output/'prediction_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'已输出{len(frame)}个样本的预测：{output}',flush=True)


def main():
    global DATA_DIR, OUT, TABLES, INFO, MODEL_DIR, SCORE_OUT, FIG
    parser = argparse.ArgumentParser(description='A题第一问：单文件训练、验证、七点评分与预测')
    parser.add_argument('--data-dir',type=Path,help='原始附件目录；默认自动寻找')
    parser.add_argument('--output',type=Path,help='结果目录；默认问题一/问题结果')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--predict',type=Path,metavar='CSV',help='预测新样本；不重新训练')
    mode.add_argument('--clean-only',action='store_true',help='仅运行完整数据清洗和核验，导出清洗数据与报告')
    parser.add_argument('--model',type=Path,help='预测使用的统一模型文件；默认读取问题一/问题结果/模型')
    args = parser.parse_args()
    default_output = QUESTION_DIR/'问题结果'/'新样本预测' if args.predict else QUESTION_DIR/'问题结果'
    OUT = (args.output or default_output).resolve()
    TABLES, INFO, MODEL_DIR = OUT/'数据表格', OUT/'元数据', OUT/'模型'
    SCORE_OUT, FIG = TABLES/'七点评分', OUT/'绘图'
    if (OUT/'3.xlsx').exists() or (OUT/'8.xlsx').exists():
        raise ValueError('输出目录不能是原始附件目录。')
    if args.predict:
        default_model = QUESTION_DIR/'问题结果'/'模型'/'question1_model.joblib'
        model_path = (args.model or default_model).resolve()
        predict_file(args.predict,model_path,OUT)
        return
    DATA_DIR = args.data_dir.resolve() if args.data_dir else None
    if DATA_DIR is not None and not all((DATA_DIR/f).is_file() for f in ['1.csv','4.csv','3.xlsx','8.xlsx']):
        raise ValueError('附件目录需要包含1.csv、4.csv、3.xlsx和8.xlsx。')
    started = time.time()
    run_cleaning()
    if args.clean_only:
        print(f'数据清洗完成：{OUT}',flush=True)
        return
    train()
    calibrate()
    combine_models()
    plot_results()
    write_report()
    verification = {'single_source_file':True,'model_serialization':'passed',
                    'fold_isolation_and_source_hashes':'passed',
                    'grade_monotonicity_and_bounds':'passed',
                    'elapsed_seconds':round(time.time()-started,2)}
    (INFO/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'全部完成：{OUT}，总耗时{verification["elapsed_seconds"]}秒',flush=True)


# Stable pickle names also support importing this file from question 2/3.
for _class in [QuantileClipper,SelectedSensoryModel,SevenPointIntensityScale]:
    _class.__module__ = 'question1'

if __name__ == '__main__':
    main()


