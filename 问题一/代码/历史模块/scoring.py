"""QDP-inspired seven-point scoring with explicitly unverified proxy anchors.

The scale format is borrowed from the source study. Numerical anchor locations
below are a project assumption, NOT concentration limits prescribed by ISO/GB.
Actual sensory calibration can replace these anchors through from_anchor_table.
"""
import numpy as np


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
