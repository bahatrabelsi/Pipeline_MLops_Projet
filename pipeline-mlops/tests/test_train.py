"""Tests unitaires des fonctions utilitaires de train.py (cible US 3.1 / CI)."""

import pandas as pd
import pytest

from src.train import load_data, train_model, evaluate


@pytest.fixture
def iris_like(tmp_path):
    """Cree un mini CSV facilement separable pour les tests."""
    df = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.2, 5.9, 5.0, 6.7],
            "sepal_width": [3.5, 3.0, 3.4, 3.0, 3.6, 3.1],
            "petal_length": [1.4, 1.4, 5.4, 5.1, 1.4, 4.7],
            "petal_width": [0.2, 0.2, 2.3, 1.8, 0.2, 1.5],
            "species": ["setosa", "setosa", "virginica",
                        "virginica", "setosa", "versicolor"],
        }
    )
    path = tmp_path / "iris.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_data_returns_features_and_target(iris_like):
    X, y = load_data(iris_like, "species")
    assert "species" not in X.columns
    assert len(X) == len(y) == 6


def test_load_data_missing_target_raises(iris_like):
    with pytest.raises(ValueError):
        load_data(iris_like, "colonne_inexistante")


def test_train_and_evaluate(iris_like):
    X, y = load_data(iris_like, "species")
    model = train_model(X, y, n_estimators=10, max_depth=3)
    accuracy, f1 = evaluate(model, X, y)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0
