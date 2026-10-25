"""Predict supplied-scale sensory attributes and estimated seven-point intensity.

Inputs must use exactly the original 1.csv representation, including mixed log10
and untransformed columns. Missing feature columns are treated as unavailable.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from data import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'new_predictions')
    parser.add_argument('--intervals', action='store_true', help='Also use the separate 150-sample split-conformal model')
    parser.add_argument('--grade-mode', choices=['reference','relative'], default='reference',
                        help='Reference-candidate seven-point scores (default), or historical quantile ranks')
    args = parser.parse_args()
    source = args.input.resolve()
    out = args.output_dir.resolve()
    # Keep inference outputs away from any supplied attachment directory.
    if (out/'3.xlsx').exists() or (out/'8.xlsx').exists():
        raise ValueError('Output directory cannot be an original attachment directory')
    bundle = joblib.load(ROOT/'models'/'sensory_model.joblib')
    frame = pd.read_csv(source,header=0,dtype={'beer':str,'beer_id':str,'tasting_category_fine':str})
    common = set(frame.columns) & set(bundle['features'])
    if not common:
        raise ValueError('No recognized chemistry feature names were found')
    X = frame.reindex(columns=bundle['features']).apply(pd.to_numeric,errors='raise').to_numpy(float)
    if np.isinf(X).any() or np.isnan(X).all(axis=1).any():
        raise ValueError('Each row needs at least one finite chemistry measurement; infinity is invalid')
    meta = frame[[c for c in ['beer','beer_id','tasting_category_fine'] if c in frame]].copy()
    meta['missing_fraction'] = np.isnan(X).mean(axis=1)
    pred = bundle['model'].predict(X)
    assert pred.shape == (len(frame),len(bundle['targets'])) and np.isfinite(pred).all()
    out.mkdir(parents=True,exist_ok=True)
    exports=[('predictions_continuous.csv',pred,bundle['targets'])]
    grade_notice=bundle['grade_notice']
    if args.grade_mode=='reference':
        scale_path=ROOT/'models'/'seven_point_scale.joblib'
        if not scale_path.exists():
            raise ValueError('Run calibrate_scores.py before reference-based intensity prediction')
        scale=joblib.load(scale_path)
        if scale['source_targets']!=bundle['targets']:
            raise ValueError('Scoring and regression target schemas do not match')
        selected=pred[:,scale['keep_mask']]
        exports.extend([
            ('estimated_intensity_scores_0_6.csv',scale['scale'].scores(selected),scale['targets']),
            ('estimated_intensity_grades_0_6.csv',scale['scale'].grades(selected),scale['targets'])])
        grade_notice=scale['notice']
    else:
        exports.append(('relative_grades_0_6.csv',bundle['model'].grades(pred),bundle['targets']))
    for name,values,targets in exports:
        target=out/name
        if target.resolve()==source:
            raise ValueError('Refusing to overwrite input')
        pd.concat([meta,pd.DataFrame(values,columns=targets)],axis=1).to_csv(target,index=False,encoding='utf-8-sig')
    notice={'score_scale':'provided standardized panel scale',
            'grade_mode':args.grade_mode,
            'grade_notice':grade_notice,
            'intensity_excludes_overall_appreciation':args.grade_mode=='reference',
            'input_notice':bundle['feature_notice'],
            'unavailable_feature_columns':sorted(set(bundle['features'])-set(frame)),
            'missing_fraction_above_stress_test_range_rows':np.flatnonzero(np.isnan(X).mean(axis=1)>0.30).tolist()}
    if args.intervals:
        b=joblib.load(ROOT/'models'/'split_conformal.joblib')
        point=b['model'].predict(X)
        for name,values in [('interval_model_point.csv',point),('interval_lower_90.csv',point-b['radius']),('interval_upper_90.csv',point+b['radius'])]:
            pd.concat([meta,pd.DataFrame(values,columns=b['targets'])],axis=1).to_csv(out/name,index=False,encoding='utf-8-sig')
        notice['interval_notice']='Per-target 90% split-conformal intervals from a separate 150-fit/50-calibration model; exchangeability and no distribution shift assumed. Not simultaneous 50-target coverage.'
    (out/'prediction_notes.json').write_text(json.dumps(notice,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {len(frame)} rows, {len(bundle["targets"])} attributes to {out}')


if __name__=='__main__':
    main()
