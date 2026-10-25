"""Verify the adopted scale implementation and output provenance, without fitting regressors."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json
import numpy as np
import pandas as pd
import joblib
from data import ROOT,load_data
from scoring import SevenPointIntensityScale


def main():
    out=ROOT/'results'/'seven_point'
    meta,X,Y,features,targets,_,_=load_data()
    bundle=joblib.load(ROOT/'models'/'seven_point_scale.joblib')
    assert bundle['targets']==[t for t in targets if t!='overall']
    p=pd.read_csv(ROOT/'results'/'oof_predictions_continuous.csv',header=0,float_precision='round_trip')
    scores=pd.read_csv(out/'oof_estimated_intensity_scores_0_6.csv',header=0,float_precision='round_trip')
    grades=pd.read_csv(out/'oof_estimated_intensity_grades_0_6.csv',header=0)
    core=pd.read_csv(out/'oof_core_aroma_flavour_grades_0_6.csv',header=0)
    assert scores.shape==(250,53) and grades.shape==(250,53) and core.shape==(250,34)
    assert list(grades.columns[4:])==bundle['targets']
    assert grades.iloc[:,4:].min().min()>=0 and grades.iloc[:,4:].max().max()<=6
    references=pd.read_csv(out/'reference_candidates_and_assumptions.csv',header=0,dtype={'reference_candidate_beer':str})
    protocol=json.loads((ROOT/'results'/'protocol.json').read_text(encoding='utf-8'))
    for fold in protocol['folds']:
        tr=np.flatnonzero(meta.beer.isin(fold['train_beer']).to_numpy())
        te=np.flatnonzero(meta.beer.isin(fold['test_beer']).to_numpy())
        fitted=SevenPointIntensityScale().fit(Y[tr][:,bundle['keep_mask']],bundle['targets'],meta.beer.iloc[tr])
        pp=p.iloc[te][bundle['targets']].to_numpy(float)
        np.testing.assert_allclose(scores.iloc[te][bundle['targets']],fitted.scores(pp),atol=1e-12)
        np.testing.assert_array_equal(grades.iloc[te][bundle['targets']],fitted.grades(pp))
        candidates=references[references.fit_scope==f'outer_train_fold_{fold["fold"]}'].reference_candidate_beer.dropna()
        assert set(candidates)<=set(fold['train_beer']) and set(candidates).isdisjoint(fold['test_beer'])
    known=pd.DataFrame({'target':['test']*3,'anchor':['low','middle','high'],
                        'provided_scale_value':[0.,3.,6.]})
    scaler=SevenPointIntensityScale.from_anchor_table(known,['test'])
    np.testing.assert_array_equal(scaler.grades(np.arange(7.).reshape(-1,1)),np.arange(7).reshape(-1,1))
    np.testing.assert_allclose(scaler.scores(np.array([[-100.],[100.]])),[[0.],[6.]])
    broken=known.copy();broken.loc[2,'provided_scale_value']=1
    try:
        SevenPointIntensityScale.from_anchor_table(broken,['test'])
    except ValueError:
        pass
    else:
        raise AssertionError('Unordered anchors were accepted')
    assert not references.sensory_calibration_confirmed.any()
    checks={'status':'passed','checks':['0-6 schema and range','30 core / 49 intensity fields',
        'outer-training-only anchors and reference IDs','OOF score recomputation',
        'known-anchor interpolation and clipping','unordered-anchor rejection',
        'unverified calibration status preserved']}
    (out/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
