import numpy as np
import pandas as pd
import pytest

from bioprimelasso.backend import BackendFit, LassoBackend


def test_backend_penalty_scaling_and_prediction():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["GeneA", "GeneB"])
    true_coef = np.array([1.5, -0.5])
    y = pd.Series(X.values @ true_coef + rng.normal(scale=0.1, size=20))

    backend = LassoBackend(max_iter=5000, tol=1e-6)
    penalties = {"GeneA": 1.0, "GeneB": 0.2}

    fit = backend.fit(X, y, lambda_=0.05, phi=0.4, penalties=penalties)

    expected_factor = pd.Series({"GeneA": 0.6, "GeneB": 0.92})
    pd.testing.assert_series_equal(fit.penalty_factor, expected_factor, check_names=False)

    preds_via_backend = backend.predict(fit, X.head())
    preds_via_fit = fit.predict(X.head())
    np.testing.assert_allclose(preds_via_backend, preds_via_fit)

    assert np.isfinite(fit.intercept)
    assert list(fit.coefficients.index) == ["GeneA", "GeneB"]


def test_backend_predict_linear_combination():
    coefficients = pd.Series({"GeneA": 2.0, "GeneB": -1.0})
    fit = BackendFit(
        coefficients=coefficients,
        intercept=0.5,
        lambda_=0.1,
        phi=0.0,
        penalty_factor=pd.Series(1.0, index=coefficients.index),
    )

    X = pd.DataFrame({"GeneA": [1.0, 0.0], "GeneB": [0.0, 2.0]})
    preds = fit.predict(X)
    np.testing.assert_allclose(preds, np.array([2.5, -1.5]))
