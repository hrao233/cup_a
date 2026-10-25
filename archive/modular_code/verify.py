"""Material consistency checks; no retraining and no network access."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score
from data import ROOT,load_data
from models import relative_grades,QuantileClipper


def main():
    out=ROOT/'results'
    meta,X,Y,features,targets,_,audit=load_data()
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    continuous=pd.read_csv(out/'oof_predictions_continuous.csv',header=0,dtype={'beer':str,'beer_id':str},float_precision='round_trip')
    grades=pd.read_csv(out/'oof_relative_grades_0_6.csv',header=0,dtype={'beer':str,'beer_id':str})
    assert list(continuous.beer)==list(meta.beer)
    assert list(continuous.columns[4:])==targets
    assert list(grades.columns[4:])==targets
    P=continuous[targets].to_numpy(float)
    np.testing.assert_allclose(r2_score(Y,P),summary['macro_r2'],atol=1e-12)
    sample_counts=np.zeros(len(X),dtype=int)
    for f in protocol['folds']:
        assert set(f['train_beer']).isdisjoint(f['test_beer'])
        tr=np.flatnonzero(meta.beer.isin(f['train_beer']).to_numpy())
        te=np.flatnonzero(meta.beer.isin(f['test_beer']).to_numpy())
        rec=joblib.load(out/f'fold_{f["fold"]}.joblib')
        np.testing.assert_array_equal(rec['te'],te)
        np.testing.assert_allclose(P[te],rec['prediction'],atol=1e-12)
        np.testing.assert_array_equal(grades[targets].to_numpy(int)[te],relative_grades(P[te],Y[tr]))
        sample_counts[te]+=1
    assert (sample_counts==1).all()
    final=joblib.load(ROOT/'models'/'sensory_model.joblib')
    assert final['features']==features and final['targets']==targets
    pred=final['model'].predict(X[:9])
    assert pred.shape==(9,50) and np.isfinite(pred).all()
    partial=X[:9].copy();partial[:,::5]=np.nan
    assert np.isfinite(final['model'].predict(partial)).all()
    b=joblib.load(ROOT/'models'/'split_conformal.joblib')
    for a,c in [('fit_beers','calibration_beers'),('fit_beers','test_beers'),('calibration_beers','test_beers')]:
        assert set(b[a]).isdisjoint(b[c])
    assert [len(b[k]) for k in ['fit_beers','calibration_beers','test_beers']]==[150,50,50]
    intervals=pd.read_csv(out/'split_conformal_test_predictions.csv',header=0,float_precision='round_trip')
    assert len(intervals)==2500 and (intervals.lower<=intervals.upper).all()
    coverage=((intervals.observed>=intervals.lower)&(intervals.observed<=intervals.upper)).mean()
    np.testing.assert_allclose(coverage,summary['split_conformal_90_mean_coverage'],atol=1e-12)
    clip=QuantileClipper().fit(np.arange(100).reshape(-1,1))
    np.testing.assert_allclose(clip.transform([[-1e6],[1e6]]),[[.99],[98.01]])
    for path in [out/'第一问分析与实验结果.md',out/'figures/question1_overview.png',out/'scorecard_beer_1_to_12_relative_grades.csv']:
        assert path.exists() and path.stat().st_size>0
    status={'status':'passed','samples':250,'targets':50,
            'checks':['CSV and array agreement','one outer test prediction per sample',
                      'grade mapping uses only outer training labels','saved model schema and missing-feature inference',
                      'split-conformal fit/calibration/test isolation and coverage recomputation',
                      'training-only clipping bounds','report and figure artifacts present']}
    (out/'verification.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
