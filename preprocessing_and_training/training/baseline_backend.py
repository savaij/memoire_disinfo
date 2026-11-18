"""Select CPU or cuML baselines and evaluate calibrated probabilities."""
import os

import numpy as np


def resolve_backend(backend=None):
    backend = backend if backend is not None else os.environ.get('BASELINE_BACKEND', 'cpu')
    if backend not in ('cpu', 'gpu'):
        raise ValueError('BASELINE_BACKEND must be cpu or gpu')
    return backend


def require_gpu():
    try:
        import cupy as cp
        import cuml
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device available')
        cp.cuda.runtime.free(0)
    except Exception as exc:
        raise RuntimeError('GPU baseline requested, but cuML/CUDA is unavailable. '
                           'Use the paper_disinfo GPU environment or select cpu.') from exc
    return cp


def build_baselines(backend=None, *, logreg_random_state=None):
    backend = resolve_backend(backend)
    if backend == 'cpu':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        models = {
            'logreg': LogisticRegression(max_iter=1000, n_jobs=-1,
                                         random_state=logreg_random_state),
            'rf': RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        }
    else:
        require_gpu()
        from cuml.linear_model import LogisticRegression
        from cuml.ensemble import RandomForestClassifier
        models = {
            'logreg': LogisticRegression(penalty='l2', C=1.0, fit_intercept=True,
                                        max_iter=1000, tol=1e-4, solver='qn',
                                        output_type='numpy'),
            'rf': RandomForestClassifier(n_estimators=250, split_criterion='gini',
                                        bootstrap=True, max_samples=1.0,
                                        max_features='sqrt', max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1,
                                        random_state=42, n_bins=128, n_streams=4,
                                        output_type='numpy'),
        }
    return models


def positive_probability(model, features):
    classes = model.classes_
    if hasattr(classes, 'get'):
        classes = classes.get()
    classes = np.asarray(classes)
    column = np.flatnonzero(classes == 1)
    if len(column) != 1:
        raise ValueError('Expected a fitted binary classifier with positive class 1')
    values = model.predict_proba(features)
    if hasattr(values, 'get'):
        values = values.get()
    values = np.asarray(values)
    if values.shape != (len(features), len(classes)):
        raise ValueError('Unexpected predict_proba shape')
    result = values[:, column[0]]
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Invalid predicted probabilities')
    return result


def fit_evaluate(model, X, y, fit, cal, test, *, backend=None, label='baseline'):
    from imbalanced_evaluation import evaluate_with_calibration
    backend = resolve_backend(backend)
    print(f'{label}: backend={backend}, fit starting', flush=True)
    model.fit(X[fit], y[fit])
    p_test = positive_probability(model, X[test])
    p_cal = positive_probability(model, X[cal])
    return evaluate_with_calibration(y[test], p_test, y[cal], p_cal)
