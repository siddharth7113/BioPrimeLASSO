"""High level orchestration of the BioPrime LASSO workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from .backend import BackendFit, LassoBackend
from .plotting import ManhattanPlotter
from .plotting.matplotlib import MatplotlibManhattanPlotter
from .repository import ResultRepository, StoredResult, create_stored_result
from .scores import ScoreProvider
from .tuning import HyperparameterTuner, TuningResult


class BioPrimeLassoModel:
    """Co-ordinates tuning, fitting, and result persistence."""

    def __init__(
        self,
        *,
        tuner: HyperparameterTuner,
        backend: LassoBackend,
        result_store: ResultRepository,
        score_provider: ScoreProvider,
        plotters: Optional[Mapping[str, ManhattanPlotter]] = None,
    ) -> None:
        self._tuner = tuner
        self._backend = backend
        self._result_store = result_store
        self._score_provider = score_provider
        self._plotters: MutableMapping[str, ManhattanPlotter] = (
            dict(plotters) if plotters is not None else {}
        )
        if "matplotlib" not in self._plotters:
            self._plotters["matplotlib"] = MatplotlibManhattanPlotter()

        self._fitted: Optional[StoredResult] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        gene: str,
        network: pd.DataFrame,
        metadata: Optional[Mapping[str, Any]] = None,
        phi_grid: Optional[Sequence[float]] = None,
        folds: int = 10,
    ) -> str:
        """Fit the BioPrime LASSO model and persist the result."""

        if phi_grid is None:
            phi_grid = np.linspace(0.0, 1.0, num=30)

        features = {name: idx for idx, name in enumerate(X.columns)}
        scores = self._score_provider.scores_for(
            gene=gene, network=network, features=features
        )

        tuning = self._tuner.tune(
            X,
            y,
            scores,
            folds=folds,
            phi_grid=phi_grid,
        )

        baseline_fit = self._backend.fit(
            X,
            y,
            lambda_=tuning.lambda_,
            phi=0.0,
            penalties=scores,
        )
        penalised_fit = self._backend.fit(
            X,
            y,
            lambda_=tuning.lambda_,
            phi=tuning.phi,
            penalties=scores,
        )

        correlations = X.apply(lambda col: col.corr(y)).fillna(0.0)

        result = create_stored_result(
            lambda_=tuning.lambda_,
            phi=tuning.phi,
            coefficients=penalised_fit.coefficients.to_dict(),
            baseline_coefficients=baseline_fit.coefficients.to_dict(),
            intercept=penalised_fit.intercept,
            correlations=correlations.to_dict(),
            metadata=metadata,
        )
        handle = self._result_store.store(result)
        self._fitted = result
        return handle

    def predict(
        self,
        X_new: pd.DataFrame,
        *,
        handle: Optional[str] = None,
    ) -> np.ndarray:
        """Generate predictions for new data."""

        result = self._resolve_result(handle)
        fit = BackendFit(
            coefficients=result.coefficients,
            intercept=result.intercept,
            lambda_=result.lambda_,
            phi=result.phi,
            penalty_factor=pd.Series(
                1.0, index=result.coefficients.index
            ),
        )
        return self._backend.predict(fit, X_new)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        gene: str,
        network: pd.DataFrame,
        phi_grid: Optional[Sequence[float]] = None,
        folds: int = 10,
    ) -> TuningResult:
        """Expose hyper-parameter search diagnostics."""

        if phi_grid is None:
            phi_grid = np.linspace(0.0, 1.0, num=30)

        features = {name: idx for idx, name in enumerate(X.columns)}
        scores = self._score_provider.scores_for(
            gene=gene, network=network, features=features
        )
        return self._tuner.tune(
            X,
            y,
            scores,
            folds=folds,
            phi_grid=phi_grid,
        )

    def plot_manhattan(
        self,
        handle: str,
        *,
        plotter: str = "matplotlib",
        output: Optional[Path | str] = None,
        highlight: Optional[Sequence[str]] = None,
    ) -> Path:
        """Render a Manhattan plot using the requested backend."""

        result = self._result_store.load(handle)
        plot_backend = self._plotters.get(plotter)
        if plot_backend is None:
            raise KeyError(f"Plotter '{plotter}' has not been registered")

        if output is None:
            output_path = Path(f"{handle}_manhattan.png")
        else:
            output_path = Path(output)

        return plot_backend.render(
            result,
            output=output_path,
            highlight=highlight,
        )

    def register_plotter(self, name: str, plotter: ManhattanPlotter) -> None:
        """Register an additional plotting backend."""

        self._plotters[name] = plotter

    def _resolve_result(self, handle: Optional[str]) -> StoredResult:
        if handle is not None:
            return self._result_store.load(handle)
        if self._fitted is None:
            raise ValueError("No model has been fitted yet")
        return self._fitted
