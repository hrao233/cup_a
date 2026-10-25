"""Apply the borrowed seven-point scale to existing OOF predictions.

This step does not retrain or re-evaluate the chemistry regression model.
Proxy anchors are selected from OUTER TRAINING labels only. No actual original
0-6 grade accuracy is reported in the absence of original grade observations.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from data import ROOT,load_data
from scoring import SevenPointIntensityScale

OUT=ROOT/'results'/'seven_point'


def save(frame,name):
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta,X,Y,features,all_targets,mapping,audit=load_data()
    base=json.loads((ROOT/'results'/'protocol.json').read_text(encoding='utf-8'))
    arrays=np.load(ROOT/'results'/'evaluation_arrays.npz')
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
    metrics=pd.read_csv(ROOT/'results'/'per_target_metrics.csv',header=0).set_index('target')
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
        ROOT/'models'/'seven_point_scale.joblib')
    restored=joblib.load(ROOT/'models'/'seven_point_scale.joblib')
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
        'continuous_oof_macro_r2':float(json.loads((ROOT/'results'/'summary.json').read_text(encoding='utf-8'))['macro_r2']),
        'verification':'passed: outer-training anchor isolation, ordinal monotonicity, endpoint/bounds and serialization checks'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
