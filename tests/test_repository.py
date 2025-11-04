from pathlib import Path

import pandas as pd

from bioprimelasso.repository import (
    InMemoryResultRepository,
    JsonResultRepository,
    StoredResult,
    create_stored_result,
)


def make_result() -> StoredResult:
    return create_stored_result(
        lambda_=0.1,
        phi=0.2,
        coefficients={"GeneA": 1.0, "GeneB": -0.5},
        baseline_coefficients={"GeneA": 0.8, "GeneB": -0.4},
        intercept=0.3,
        correlations={"GeneA": 0.6, "GeneB": -0.2},
        metadata={"chromosome": {"GeneA": "1", "GeneB": "2"}},
        handle="fixed",
    )


def test_in_memory_repository_round_trip():
    repo = InMemoryResultRepository()
    result = make_result()

    handle = repo.store(result)
    retrieved = repo.load(handle)

    assert handle == "fixed"
    pd.testing.assert_series_equal(retrieved.coefficients, result.coefficients)
    pd.testing.assert_series_equal(
        retrieved.baseline_coefficients, result.baseline_coefficients
    )
    assert retrieved.metadata == result.metadata


def test_json_repository_persists_payload(tmp_path: Path):
    repo = JsonResultRepository(tmp_path)
    result = make_result()

    handle = repo.store(result)
    path = tmp_path / f"{handle}.json"
    assert path.exists()

    loaded = repo.load(handle)
    pd.testing.assert_series_equal(loaded.coefficients, result.coefficients)
    pd.testing.assert_series_equal(
        loaded.baseline_coefficients, result.baseline_coefficients
    )
    assert loaded.metadata == result.metadata
