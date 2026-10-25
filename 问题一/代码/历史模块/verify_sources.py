"""Verify that original-author supplementary S1/S4 reproduce local inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import openpyxl
from data import ROOT, attachment_dir, sheet_frame


def main():
    path=ROOT/'external_sources'/'Supplemental Files and Figure source files.xlsx'
    wb=openpyxl.load_workbook(path,read_only=True,data_only=True)
    checks={}
    for name,filename in [('Supplementary File S1','1.csv'),('Supplementary File S4','4.csv')]:
        original=sheet_frame(wb,name)
        local=pd.read_csv(attachment_dir()/filename,header=0,dtype={'beer_id':str})
        assert original.shape==local.shape and list(original.columns)==list(local.columns)
        assert list(original.beer.astype(str))==list(local.beer.astype(str))
        a=original.iloc[:,3:].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        b=local.iloc[:,3:].to_numpy(float)
        assert np.array_equal(np.isnan(a),np.isnan(b))
        delta=float(np.nanmax(np.abs(a-b)))
        assert delta<1e-12
        checks[name]={'shape':list(original.shape),'numeric_columns_compared':len(original.columns)-3,
                      'max_abs_difference':delta,'same_missing_mask':True,'join_key':'beer',
                      'beer_id_note':'excluded from numeric equality check because local CSV may round scientific notation'}
    wb.close()
    script=(ROOT/'external_sources'/'Sensory data quality control - ANOVA on repeated samples.Rmd').read_text(encoding='utf-8')
    assert 'Raw_Sensory_data_Trained_panel.csv' in script
    manifest=json.loads((ROOT/'external_sources'/'manifest.json').read_text(encoding='utf-8'))
    assert not any('Raw_Sensory_data_Trained_panel.csv'==r['file'] for r in manifest)
    result={'source':'https://zenodo.org/records/10653704',
            'workbook':path.name,'checked_sheets':checks,
            'raw_panelist_scores':'referenced by original QC script; not present among downloaded public supplementary files',
            'note':'external copies are for provenance checks, not extra training observations'}
    (ROOT/'results'/'source_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
