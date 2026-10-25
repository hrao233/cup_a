"""Candidate models, nested selection, and explicit provisional grade mapping."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold


def candidates(seed=20261024):
    models = {}
    for alpha in [10., 100., 1000.]:
        models[f'ridge_{int(alpha)}'] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=alpha))
    for n in [5, 12]:
        models[f'pls_{n}'] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), PLSRegression(n_components=n, scale=False, max_iter=1000))
    for leaf in [2, 6]:
        tree = ExtraTreesRegressor(n_estimators=220, min_samples_leaf=leaf, max_features=0.7, random_state=seed, n_jobs=2)
        models[f'extra_trees_leaf{leaf}'] = make_pipeline(SimpleImputer(strategy='median'), tree)
    models['hist_gradient_boost'] = make_pipeline(
        SimpleImputer(strategy='median'),
        MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=7,
            max_bins=32, learning_rate=0.06, min_samples_leaf=12, l2_regularization=5,
            early_stopping=False, random_state=seed), n_jobs=2))
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
        blend_ix = [self.names_.index(n) for n in ['ridge_100', 'extra_trees_leaf2', 'hist_gradient_boost']]
        self.cv_predictions_ = np.concatenate([self.cv_predictions_, self.cv_predictions_[blend_ix].mean(axis=0)[None]], axis=0)
        self.names_.append('blend_tree_linear')
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
        predictions.append(np.mean([predictions[base_names.index(n)]
                                    for n in ['ridge_100', 'extra_trees_leaf2', 'hist_gradient_boost']], axis=0))
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
