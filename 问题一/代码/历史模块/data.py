"""Read-only input audit and explicit schema construction for question 1."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
META = ['beer', 'beer_id', 'tasting_category_fine']


def attachment_dir():
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
        (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return chem[META].copy(), X, Y, features, targets, mapping, audit


if __name__ == '__main__':
    *_, audit = load_data(ROOT / 'results')
    print(json.dumps(audit, ensure_ascii=False, indent=2))
