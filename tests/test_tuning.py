import numpy as np
import pandas as pd
import pytest

from bioprimelasso.tuning import (
    GridSearchTuner,
    _compute_rmse_matrix,
    _select_phi_from_rmse,
)


def make_toy_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["GeneA", "GeneB", "GeneC"])
    coef = np.array([1.0, -2.0, 0.5])
    y = pd.Series(X.values @ coef + rng.normal(scale=0.1, size=40))
    scores = {"GeneA": 0.9, "GeneB": 0.2, "GeneC": 0.0}
    return X, y, scores


def test_compute_rmse_matrix_shape_and_standardisation():
    X, y, scores = make_toy_data()
    phi_values = np.linspace(0.0, 0.5, num=4)
    rmse = _compute_rmse_matrix(
        X,
        y,
        scores,
        lambda_value=0.05,
        phi_values=phi_values,
        folds=4,
        random_state=0,
    )

    assert list(rmse.index) == list(phi_values)
    assert rmse.shape == (4, 4)
    column_means = rmse.mean(axis=0).round(7)
    assert all(abs(mean) < 1e-6 for mean in column_means)


def test_select_phi_from_rmse_prefers_lowest_detrended_value():
    rmse = pd.DataFrame(
        {
            0: [0.5, 0.6, 0.7],
            1: [0.6, 0.7, 0.9],
        },
        index=[0.0, 0.5, 1.0],
    )
    phi = _select_phi_from_rmse(rmse, rmse.index)
    assert phi == pytest.approx(0.5)


def test_grid_search_tuner_returns_valid_result():
    X, y, scores = make_toy_data()
    tuner = GridSearchTuner(random_state=0)
    phi_grid = np.linspace(0.0, 0.5, num=5)

    result = tuner.tune(X, y, scores, folds=4, phi_grid=phi_grid)

    assert result.lambda_ > 0
    assert result.phi in phi_grid
    assert isinstance(result.rmse, pd.DataFrame)
