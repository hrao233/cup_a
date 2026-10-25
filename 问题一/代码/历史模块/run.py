"""Reproducible nested CV, missing-feature stress tests, calibration, final model.

No internet access is needed. Source attachments are never modified.
All reported generalization metrics use unseen outer-fold observations.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_key] = '1'

from pathlib import Path
import argparse
import json
import time
import platform
import warnings
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from data import ROOT, load_data
from models import SelectedSensoryModel, relative_grades, z_anchor_grades

SEED = 20261024
OUT = ROOT / 'results'


def csv(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def score(y, p):
    return {'macro_r2': float(r2_score(y, p, multioutput='uniform_average')),
            'mae_provided_scale': float(np.abs(y-p).mean()),
            'rmse_provided_scale': float(np.sqrt(np.mean((y-p)**2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Reuse completed outer-fold checkpoints')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'models').mkdir(exist_ok=True)
    started = time.time()
    meta, X, Y, features, targets, mapping, audit = load_data(OUT)
    n, q = Y.shape
    core = np.array([t.startswith(('A_', 'F_')) for t in targets])
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    test_count = np.zeros(n, dtype=int)
    for tr, te in folds:
        assert not set(tr) & set(te)
        test_count[te] += 1
    assert (test_count == 1).all()
    (OUT / 'protocol.json').write_text(json.dumps({
        'seed': SEED, 'outer_splits': 5, 'inner_splits': 3,
        'target_count': 50, 'primary_core_target_count': 30,
        'model_selection': 'minimum inner-fold OOF MSE independently per target',
        'excluded_predictors': ['beer', 'beer_id', 'tasting_category_fine', '4.csv', '7.csv', 'external model outputs'],
        'missingness_scenarios': [0.0, 0.1, 0.2, 0.3], 'mask_repeats': 5,
        'stress_test': 'additional MCAR masking of observed outer-test feature cells; no train/test refit',
        'grade_mapping': 'seven train-label quantile bins, grade 0 is a relative rank, not absence',
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
        checkpoint = OUT / f'fold_{fold}.joblib'
        if args.resume and checkpoint.exists():
            rec = joblib.load(checkpoint)
            assert np.array_equal(rec['te'], te)
            print(f'Fold {fold}: checkpoint restored', flush=True)
        else:
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
            joblib.dump(rec, checkpoint)
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
    np.savez_compressed(OUT/'evaluation_arrays.npz',y=Y,prediction=oof,baseline=baseline,grades=grades,fold=fold_labels)

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
                     test_beers=meta.beer.iloc[test].tolist()),ROOT/'models'/'split_conformal.joblib')

    print('Fitting reusable model on all 250 samples; these fitted values are NOT evaluation results...',flush=True)
    final=SelectedSensoryModel(seed=SEED).fit(X,Y)
    joblib.dump(dict(model=final,features=features,targets=targets,seed=SEED,
                     grade_notice='provisional relative quantile bins; not recovered original 0-6 scores',
                     feature_notice='use exactly the input CSV representation; no second log transform'),
                ROOT/'models'/'sensory_model.joblib')
    csv(pd.DataFrame({'target':targets,'selected_model':[final.names_[j] for j in final.selected_]}),'final_model_selection.csv')
    csv(pd.DataFrame(final.inner_mse_.T,columns=final.names_,index=targets).reset_index(names='target'),'final_inner_cv_mse.csv')
    csv(pd.DataFrame(final.y_reference_,columns=targets),'grade_reference_distribution.csv')
    # Serialization, input/output alignment, and finite predictions are material checks.
    restored=joblib.load(ROOT/'models'/'sensory_model.joblib')
    np.testing.assert_allclose(final.predict(X[:7]),restored['model'].predict(X[:7]),rtol=0,atol=1e-12)
    assert np.isfinite(oof).all() and np.isfinite(final.predict(X)).all()
    assert grades.min()>=0 and grades.max()<=6
    for entry in audit['input_manifest']:
        import hashlib
        from data import attachment_dir
        assert hashlib.sha256((attachment_dir()/entry['file']).read_bytes()).hexdigest()==entry['sha256']
    result['elapsed_seconds']=round(time.time()-started,2)
    result['verification']='passed: nested fold isolation, finite predictions, grade bounds, serialization, unchanged source hashes'
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
