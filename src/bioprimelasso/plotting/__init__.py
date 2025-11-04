"""Plotting backends for BioPrime LASSO."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Sequence

from ..repository import StoredResult


class ManhattanPlotter(Protocol):
    """Protocol describing the required plotting interface."""

    def render(
        self,
        result: StoredResult,
        *,
        output: Path,
        highlight: Optional[Sequence[str]] = None,
    ) -> Path:
        ...
