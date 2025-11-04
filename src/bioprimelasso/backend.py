"""Backend fitting utilities built on top of scikit-learn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso


@dataclass
class BackendFit:
    """Encapsulates a fitted LASSO model."""

    coefficients: pd.Series
    intercept: float
    lambda_: float
    phi: float
    penalty_factor: pd.Series

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X @ self.coefficients.values + self.intercept)


class LassoBackend:
    """Fit LASSO models while respecting feature-specific penalties."""

    def __init__(self, *, max_iter: int = 10_000, tol: float = 1e-4) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        lambda_: float,
        phi: float,
        penalties: Mapping[str, float],
    ) -> BackendFit:
        feature_names = X.columns
        penalty = np.array([penalties.get(name, 0.0) for name in feature_names])
        penalty_factor = 1.0 - penalty * phi
        penalty_factor = np.clip(penalty_factor, 1e-6, None)

        scaled_X = X / penalty_factor

        model = Lasso(alpha=lambda_, max_iter=self.max_iter, tol=self.tol)
        model.fit(scaled_X, y)

        coefficients = pd.Series(model.coef_ / penalty_factor, index=feature_names)
        penalty_series = pd.Series(penalty_factor, index=feature_names)
        return BackendFit(
            coefficients=coefficients,
            intercept=float(model.intercept_),
            lambda_=lambda_,
            phi=phi,
            penalty_factor=penalty_series,
        )

    def predict(self, fit: BackendFit, X: pd.DataFrame) -> np.ndarray:
        return fit.predict(X)
