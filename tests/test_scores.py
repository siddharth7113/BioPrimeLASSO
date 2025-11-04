import pandas as pd
import pytest

from bioprimelasso.scores import NetworkScoreProvider


def test_network_score_provider_requires_columns():
    provider = NetworkScoreProvider()
    network = pd.DataFrame({"gene1": ["A"], "gene2": ["B"]})

    with pytest.raises(ValueError):
        provider.scores_for(gene="A", network=network, features={"A": 0})


def test_network_score_provider_normalises_scores():
    provider = NetworkScoreProvider()
    network = pd.DataFrame(
        {
            "gene1": ["TP53", "TP53", "MDM2"],
            "gene2": ["MDM2", "BRCA1", "TP53"],
            "combined_score": [900, 450, 300],
        }
    )
    features = {"TP53": 0, "MDM2": 1, "BRCA1": 2, "EGFR": 3}

    scores = provider.scores_for(gene="TP53", network=network, features=features)

    assert set(scores) == set(features)
    assert scores["TP53"] == pytest.approx(1.0)
    assert scores["MDM2"] == pytest.approx(1.0)
    assert scores["BRCA1"] == pytest.approx(0.5)
    assert scores["EGFR"] == pytest.approx(0.0)
    assert all(0.0 <= value <= 1.0 for value in scores.values())
