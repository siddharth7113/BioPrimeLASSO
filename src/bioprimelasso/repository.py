"""Storage utilities for fitted BioPrime LASSO models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional
import json
import uuid

import pandas as pd


@dataclass
class StoredResult:
    """Container for serialized model outputs."""

    handle: str
    lambda_: float
    phi: float
    coefficients: pd.Series
    baseline_coefficients: pd.Series
    intercept: float
    feature_names: Iterable[str]
    correlations: Optional[pd.Series] = None
    metadata: Optional[Mapping[str, object]] = None


class ResultRepository:
    """Abstract repository for storing and retrieving model outputs."""

    def store(self, result: StoredResult) -> str:
        raise NotImplementedError

    def load(self, handle: str) -> StoredResult:
        raise NotImplementedError


class InMemoryResultRepository(ResultRepository):
    """Simple in-memory repository suitable for exploratory workflows."""

    def __init__(self) -> None:
        self._storage: MutableMapping[str, StoredResult] = {}

    def store(self, result: StoredResult) -> str:  # type: ignore[override]
        self._storage[result.handle] = result
        return result.handle

    def load(self, handle: str) -> StoredResult:  # type: ignore[override]
        return self._storage[handle]


class JsonResultRepository(ResultRepository):
    """Persist fitted models to disk as JSON payloads."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def store(self, result: StoredResult) -> str:  # type: ignore[override]
        handle = result.handle or uuid.uuid4().hex
        payload = {
            "handle": handle,
            "lambda": result.lambda_,
            "phi": result.phi,
            "coefficients": result.coefficients.to_dict(),
            "baseline_coefficients": result.baseline_coefficients.to_dict(),
            "intercept": result.intercept,
            "feature_names": list(result.feature_names),
            "correlations": None
            if result.correlations is None
            else result.correlations.to_dict(),
            "metadata": result.metadata or {},
        }
        path = self.directory / f"{handle}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return handle

    def load(self, handle: str) -> StoredResult:  # type: ignore[override]
        path = self.directory / f"{handle}.json"
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return StoredResult(
            handle=payload["handle"],
            lambda_=payload["lambda"],
            phi=payload["phi"],
            coefficients=pd.Series(payload["coefficients"]),
            baseline_coefficients=pd.Series(payload["baseline_coefficients"]),
            intercept=payload["intercept"],
            feature_names=payload["feature_names"],
            correlations=None
            if payload.get("correlations") is None
            else pd.Series(payload["correlations"]),
            metadata=payload.get("metadata"),
        )


def create_stored_result(
    *,
    lambda_: float,
    phi: float,
    coefficients: Mapping[str, float],
    baseline_coefficients: Mapping[str, float],
    intercept: float,
    correlations: Optional[Mapping[str, float]] = None,
    metadata: Optional[Mapping[str, object]] = None,
    handle: Optional[str] = None,
) -> StoredResult:
    """Factory to standardise :class:`StoredResult` instantiation."""

    handle_value = handle or uuid.uuid4().hex
    coeff_series = pd.Series(coefficients)
    baseline_series = pd.Series(baseline_coefficients)
    corr_series = None if correlations is None else pd.Series(correlations)
    return StoredResult(
        handle=handle_value,
        lambda_=lambda_,
        phi=phi,
        coefficients=coeff_series,
        baseline_coefficients=baseline_series,
        intercept=intercept,
        feature_names=coeff_series.index.tolist(),
        correlations=corr_series,
        metadata=dict(metadata) if metadata is not None else None,
    )
