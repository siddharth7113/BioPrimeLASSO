"""Hyper-parameter tuning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold


@dataclass
class CrossValidationReport:
    """Container for cross-validation diagnostics."""

    phi_values: np.ndarray
    lambda_value: float
    rmse_matrix: pd.DataFrame


@dataclass
class TuningResult:
    """Output from the hyper-parameter tuner."""

    lambda_: float
    phi: float
    rmse: pd.DataFrame


class HyperparameterTuner(Protocol):
    """Protocol for classes that can tune BioPrime LASSO hyper-parameters."""

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scores: Mapping[str, float],
        *,
        folds: int,
        phi_grid: Sequence[float],
    ) -> TuningResult:
        ...

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scores: Mapping[str, float],
        *,
        folds: int,
        phi_grid: Sequence[float],
    ) -> CrossValidationReport:
        ...


class GridSearchTuner:
    """Replicates the R workflow using scikit-learn primitives."""

    def __init__(self, *, random_state: int | None = None) -> None:
        self.random_state = random_state

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scores: Mapping[str, float],
        *,
        folds: int,
        phi_grid: Sequence[float],
    ) -> TuningResult:  # type: ignore[override]
        cv_report = self.cross_validate(
            X,
            y,
            scores,
            folds=folds,
            phi_grid=phi_grid,
        )
        phi = _select_phi_from_rmse(cv_report.rmse_matrix, cv_report.phi_values)
        return TuningResult(
            lambda_=cv_report.lambda_value,
            phi=float(phi),
            rmse=cv_report.rmse_matrix,
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scores: Mapping[str, float],
        *,
        folds: int,
        phi_grid: Sequence[float],
    ) -> CrossValidationReport:  # type: ignore[override]
        lambda_value = _find_lambda(X, y, folds)
        rmse_matrix = _compute_rmse_matrix(
            X,
            y,
            scores,
            lambda_value=lambda_value,
            phi_values=np.asarray(phi_grid, dtype=float),
            folds=folds,
            random_state=self.random_state,
        )
        return CrossValidationReport(
            phi_values=np.asarray(phi_grid, dtype=float),
            lambda_value=lambda_value,
            rmse_matrix=rmse_matrix,
        )


def _find_lambda(X: pd.DataFrame, y: pd.Series, folds: int) -> float:
    model = LassoCV(cv=folds, random_state=None)
    model.fit(X, y)
    return float(model.alpha_)


def _compute_rmse_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    scores: Mapping[str, float],
    *,
    lambda_value: float,
    phi_values: np.ndarray,
    folds: int,
    random_state: int | None,
) -> pd.DataFrame:
    feature_names = X.columns
    penalties = np.array([scores.get(name, 0.0) for name in feature_names])
    if penalties.max() > 0:
        penalties = penalties / penalties.max()

    kfold = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    rmse = np.zeros((len(phi_values), folds))

    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for phi_index, phi in enumerate(phi_values):
            penalty_factor = 1.0 - penalties * phi
            penalty_factor = np.clip(penalty_factor, 1e-6, None)

            X_train_scaled = X_train / penalty_factor
            X_test_scaled = X_test / penalty_factor

            model = Lasso(alpha=lambda_value, max_iter=10_000)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            rmse[phi_index, fold_index] = np.sqrt(np.mean((y_test - preds) ** 2))

    rmse_df = pd.DataFrame(rmse, index=phi_values)
    rmse_df = rmse_df.apply(_zscore_column, axis=0)
    return rmse_df


def _zscore_column(col: pd.Series) -> pd.Series:
    std = col.std(ddof=0)
    if std == 0:
        return col - col.mean()
    return (col - col.mean()) / std


def _select_phi_from_rmse(rmse: pd.DataFrame, phi_values: Iterable[float]) -> float:
    phi_array = np.asarray(list(phi_values), dtype=float)
    medians = rmse.median(axis=1).values
    if len(phi_array) < 2:
        return float(phi_array[0])
    endpoints = pd.DataFrame(
        {
            "phi": [phi_array[0], phi_array[-1]],
            "rmse": [medians[0], medians[-1]],
        }
    )
    slope = (endpoints.loc[1, "rmse"] - endpoints.loc[0, "rmse"]) / (
        endpoints.loc[1, "phi"] - endpoints.loc[0, "phi"]
    )
    intercept = endpoints.loc[0, "rmse"] - slope * endpoints.loc[0, "phi"]
    detrended = medians - (slope * phi_array + intercept)
    best_index = int(np.argmin(detrended))
    return float(phi_array[best_index])
