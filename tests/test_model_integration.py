import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from bioprimelasso.backend import LassoBackend
from bioprimelasso.model import BioPrimeLassoModel
from bioprimelasso.repository import InMemoryResultRepository
from bioprimelasso.scores import NetworkScoreProvider
from bioprimelasso.tuning import GridSearchTuner


def make_integration_inputs():
    rng = np.random.default_rng(123)
    X = pd.DataFrame(
        rng.normal(size=(60, 3)),
        columns=["GeneA", "GeneB", "GeneC"],
    )
    coef = np.array([1.2, -1.8, 0.0])
    y = pd.Series(X.values @ coef + rng.normal(scale=0.2, size=len(X)))

    network = pd.DataFrame(
        {
            "gene1": ["TP53", "TP53", "TP53"],
            "gene2": ["GeneA", "GeneB", "GeneC"],
            "combined_score": [900, 600, 0],
        }
    )
    metadata = {
        "chromosome": {"GeneA": "1", "GeneB": "3", "GeneC": "7"},
        "position": {"GeneA": 10_000_000, "GeneB": 20_000_000, "GeneC": 30_000_000},
    }
    return X, y, network, metadata


def test_bioprime_lasso_model_end_to_end(tmp_path):
    X, y, network, metadata = make_integration_inputs()
    tuner = GridSearchTuner(random_state=0)
    backend = LassoBackend(max_iter=5000, tol=1e-5)
    repository = InMemoryResultRepository()
    score_provider = NetworkScoreProvider()
    model = BioPrimeLassoModel(
        tuner=tuner,
        backend=backend,
        result_store=repository,
        score_provider=score_provider,
    )

    phi_grid = np.linspace(0.0, 0.6, num=4)
    handle = model.fit(
        X,
        y,
        gene="TP53",
        network=network,
        metadata=metadata,
        phi_grid=phi_grid,
        folds=4,
    )

    assert isinstance(handle, str)
    stored = repository.load(handle)
    assert stored.lambda_ > 0
    assert stored.phi in phi_grid
    assert not stored.coefficients.empty

    predictions = model.predict(X, handle=handle)
    assert predictions.shape == (len(X),)
    mae = np.mean(np.abs(predictions - y.values))
    assert mae < 0.6

    cv_result = model.cross_validate(
        X,
        y,
        gene="TP53",
        network=network,
        phi_grid=phi_grid,
        folds=4,
    )
    assert cv_result.lambda_ > 0
    assert cv_result.phi in phi_grid

    plot_path = model.plot_manhattan(handle, output=tmp_path / "manhattan.png")
    assert plot_path.exists()

    # Ensure predict defaults to most recent fit when handle omitted
    default_predictions = model.predict(X)
    np.testing.assert_allclose(default_predictions, predictions)
