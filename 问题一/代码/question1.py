"""A题第一问：单文件完整程序。

训练与输出：python "问题一/代码/question1.py"
指定附件：  python "问题一/代码/question1.py" --data-dir "附件文件夹"
新样本预测：python "问题一/代码/question1.py" --predict "新样本.csv"

包含：数据审计、训练折内预处理、5×3嵌套交叉验证、多模型比较、
缺失压力实验、独立区间实验、QDP七点估计评分、图表、报告及模型保存。
只依赖原始附件和常用第三方包，不导入本项目其他Python文件，不需要联网。

注意：七点评分的经验参照锚点是建模假设，并非已校准的历史人工评分。
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
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
from scipy.stats import spearmanr
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
    frame = pd.read_csv(path, header=0, dtype={c: 'string' for c in META})
    assert frame.shape == expected_shape, (path.name, frame.shape, expected_shape)
    assert not frame['beer'].isna().any()
    assert frame['beer'].is_unique
    # Beer IDs stored in rounded scientific notation are not valid join keys.
    return frame


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

    calibrated = sheet_frame(wb8, 'Calibrated chemical dataset')
    calibrated['beer'] = calibrated.beer.astype(str)
    calibrated = calibrated.set_index('beer').loc[chem.beer]
    scales = []
    for f in features:
        item = {'feature': f, 'representation': 'provided_scale_unresolved', 'checked_pairs': 0}
        if f in calibrated.columns:
            raw = pd.to_numeric(calibrated[f], errors='coerce').to_numpy(float)
            obs = chem[f].to_numpy(float)
            keep = np.isfinite(raw) & np.isfinite(obs) & (raw > 0)
            e_log = float(np.median(np.abs(np.log10(raw[keep]) - obs[keep])))
            e_raw = float(np.median(np.abs(raw[keep] - obs[keep])))
            item.update(checked_pairs=int(keep.sum()), median_log10_error=e_log, median_raw_error=e_raw,
                        representation='log10' if e_log < 1e-7 else 'raw' if e_raw < 1e-7 else 'unresolved')
        scales.append(item)
    scales = pd.DataFrame(scales)
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
    parser.add_argument('--predict',type=Path,metavar='CSV',help='预测新样本；不重新训练')
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


