"""Tests unitaires du detecteur de derive (MLOPSB-14)."""

import numpy as np
import pandas as pd

from src.data_drift import compute_psi, analyse_drift


def test_psi_nul_si_meme_distribution():
    """Deux echantillons identiques -> PSI proche de 0."""
    rng = np.random.default_rng(42)
    data = pd.Series(rng.normal(0, 1, 1000))
    psi = compute_psi(data, data)
    assert psi < 0.01


def test_psi_eleve_si_distribution_decalee():
    """Un fort decalage de moyenne -> PSI eleve."""
    rng = np.random.default_rng(42)
    baseline = pd.Series(rng.normal(0, 1, 1000))
    current = pd.Series(rng.normal(5, 1, 1000))   # moyenne decalee
    psi = compute_psi(baseline, current)
    assert psi > 0.25


def test_analyse_drift_detecte_la_derive():
    rng = np.random.default_rng(0)
    baseline = pd.DataFrame({
        "f1": rng.normal(0, 1, 500),
        "species": ["a"] * 500,
    })
    current = pd.DataFrame({
        "f1": rng.normal(4, 1, 500),
        "species": ["a"] * 500,
    })
    results, drift = analyse_drift(baseline, current, ["f1"], threshold=0.25)
    assert drift is True
    assert results["f1"]["drift"] is True
