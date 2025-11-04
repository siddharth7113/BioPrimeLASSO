"""Utilities for obtaining biological prior scores."""
from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np
import pandas as pd


class ScoreProvider(Protocol):
    """Protocol describing objects capable of producing feature scores."""

    def scores_for(
        self, *, gene: str, network: pd.DataFrame, features: Mapping[str, int]
    ) -> Mapping[str, float]:
        """Return association scores for the requested features."""


class NetworkScoreProvider:
    """Generate scores using a STRING-like interaction network."""

    def scores_for(
        self, *, gene: str, network: pd.DataFrame, features: Mapping[str, int]
    ) -> Mapping[str, float]:  # type: ignore[override]
        required_cols = {"gene1", "gene2", "combined_score"}
        if not required_cols.issubset(network.columns):
            raise ValueError(
                "Network data frame must contain columns gene1, gene2, combined_score"
            )

        mask = (network["gene1"] == gene) | (network["gene2"] == gene)
        subset = network.loc[mask, ["gene1", "gene2", "combined_score"]]
        if subset.empty:
            return {feature: 0.0 for feature in features}

        scores: dict[str, float] = {}
        for _, row in subset.iterrows():
            partner = row["gene2"] if row["gene1"] == gene else row["gene1"]
            value = float(row["combined_score"])
            existing = scores.get(str(partner), 0.0)
            scores[str(partner)] = max(existing, value)
        scores[gene] = max(scores.values(), default=1.0)

        # Align to requested features and normalise to [0, 1]
        aligned = np.array([scores.get(str(f), 0.0) for f in features])
        if aligned.max() > 0:
            aligned = aligned / aligned.max()
        return {str(feature): float(value) for feature, value in zip(features, aligned)}
