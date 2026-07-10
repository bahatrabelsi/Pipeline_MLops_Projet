"""
Tests unitaires du script de scoring - Ticket MLOPSB-17.
On teste la logique de parsing du payload SANS charger le vrai modele
(on injecte un modele factice), ce qui rend les tests rapides et sans Azure.
"""

import json

import pandas as pd
import pytest

import src.score as score


class FakeModel:
    """Modele factice : renvoie une prediction par ligne recue."""

    def predict(self, df):
        return pd.Series(["setosa"] * len(df)).values


@pytest.fixture(autouse=True)
def inject_fake_model(monkeypatch):
    monkeypatch.setattr(score, "model", FakeModel())


def test_run_payload_valide():
    payload = json.dumps({"data": [
        {"sepal_length": 5.1, "sepal_width": 3.5,
         "petal_length": 1.4, "petal_width": 0.2}
    ]})
    result = score.run(payload)
    assert result["predictions"] == ["setosa"]


def test_run_cle_data_manquante():
    """Un payload sans la cle 'data' ne doit PAS faire planter le service."""
    result = score.run(json.dumps({"input": []}))
    assert "error" in result


def test_run_json_invalide():
    """Un JSON malforme est intercepte proprement (pas d'erreur 500)."""
    result = score.run("ceci n'est pas du json")
    assert "error" in result


def test_run_colonne_manquante():
    """Une feature manquante renvoie une erreur explicite."""
    payload = json.dumps({"data": [{"sepal_length": 5.1}]})
    result = score.run(payload)
    assert "error" in result
