"""Matplotlib implementation of Manhattan plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import ManhattanPlotter
from ..repository import StoredResult


class MatplotlibManhattanPlotter(ManhattanPlotter):
    """Render Manhattan-style scatter plots of model results."""

    def __init__(self, *, dpi: int = 150) -> None:
        self.dpi = dpi

    def render(
        self,
        result: StoredResult,
        *,
        output: Path,
        highlight: Optional[Sequence[str]] = None,
    ) -> Path:
        output_path = Path(output)
        frame = _prepare_frame(result)
        highlight_set = set(highlight or [])

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        chromosomes = frame["chromosome"].unique()
        palette = dict(
            zip(chromosomes, sns.color_palette("husl", len(chromosomes)))
        )

        for chrom, subset in frame.groupby("chromosome"):
            ax.scatter(
                subset["position"],
                subset["correlation"],
                s=20,
                color=palette[chrom],
                alpha=0.7,
                label=str(chrom),
            )

        if highlight_set:
            highlight_points = frame[frame["gene"].isin(highlight_set)]
            ax.scatter(
                highlight_points["position"],
                highlight_points["correlation"],
                s=80,
                edgecolor="black",
                facecolor="none",
                linewidth=1.5,
            )
            for _, row in highlight_points.iterrows():
                ax.text(
                    row["position"],
                    row["correlation"],
                    row["gene"],
                    fontsize=9,
                    ha="right",
                    va="bottom",
                )

        ax.set_xlabel("Genomic position")
        ax.set_ylabel("Correlation with dependency score")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("BioPrime LASSO Manhattan plot")
        ax.legend(title="Chromosome", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return output_path


def _prepare_frame(result: StoredResult) -> pd.DataFrame:
    coefficients = result.coefficients.rename("beta_penalised")
    baseline = result.baseline_coefficients.rename("beta_baseline")
    frame = pd.concat([coefficients, baseline], axis=1)
    frame["gene"] = frame.index

    if result.correlations is not None:
        frame["correlation"] = result.correlations.reindex(frame.index).fillna(0.0)
    else:
        frame["correlation"] = 0.0

    metadata = result.metadata or {}
    chromosome = pd.Series(metadata.get("chromosome"), index=frame.index)
    position = pd.Series(metadata.get("position"), index=frame.index)

    if chromosome.isnull().all():
        chromosome = pd.Series("chr0", index=frame.index)
    if position.isnull().all():
        position = pd.Series(np.arange(len(frame)), index=frame.index)

    frame["chromosome"] = chromosome.astype(str)
    frame["position"] = position.astype(float)
    frame.sort_values(["chromosome", "position"], inplace=True)
    return frame
