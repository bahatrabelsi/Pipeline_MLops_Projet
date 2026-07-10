"""
Detection de derive des donnees (Data Drift) - Ticket MLOPSB-14.

Compare la distribution des donnees de reference (celles ayant servi a
l'entrainement) avec celle des nouvelles donnees observees en production.

Metrique utilisee : PSI (Population Stability Index), standard industriel.
    PSI < 0.10  -> pas de derive significative
    0.10 - 0.25 -> derive moderee (a surveiller)
    PSI > 0.25  -> derive importante -> re-entrainement recommande

Sortie : code retour 0 (pas de derive) ou 1 (derive detectee).
Ce code retour est exploite par le workflow de re-entrainement (MLOPSB-15).
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import mlflow

# Seuil au-dela duquel on considere qu'il y a derive
PSI_THRESHOLD = 0.25
EPSILON = 1e-6  # evite les divisions par zero


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True,
                        help="CSV de reference (donnees d'entrainement)")
    parser.add_argument("--current", type=str, required=True,
                        help="CSV des nouvelles donnees (production)")
    parser.add_argument("--target_column", type=str, default="species")
    parser.add_argument("--threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument("--output", type=str, default="drift_report.json")
    return parser.parse_args()


def compute_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Calcule le PSI d'une variable numerique entre deux echantillons.

    On decoupe la distribution de reference en quantiles, puis on compare
    la proportion d'observations tombant dans chaque intervalle.
    """
    # Bornes basees sur les quantiles de la reference
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    # Ouvrir les bornes pour capturer les valeurs hors intervalle
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    base_pct = base_counts / max(len(baseline), 1) + EPSILON
    curr_pct = curr_counts / max(len(current), 1) + EPSILON

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)


def analyse_drift(baseline_df, current_df, features, threshold):
    """Calcule le PSI de chaque feature et determine s'il y a derive."""
    results = {}
    for col in features:
        psi = compute_psi(baseline_df[col], current_df[col])
        results[col] = {
            "psi": round(psi, 4),
            "drift": bool(psi > threshold),
        }
    drift_detected = any(v["drift"] for v in results.values())
    return results, drift_detected


def main():
    args = parse_args()

    baseline_df = pd.read_csv(args.baseline)
    current_df = pd.read_csv(args.current)

    # On ne compare que les features (pas la cible)
    features = [c for c in baseline_df.columns if c != args.target_column]

    missing = [c for c in features if c not in current_df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes des donnees courantes : {missing}")

    results, drift_detected = analyse_drift(
        baseline_df, current_df, features, args.threshold
    )

    report = {
        "threshold": args.threshold,
        "drift_detected": drift_detected,
        "features": results,
        "n_baseline": len(baseline_df),
        "n_current": len(current_df),
    }

    # Journalisation dans MLflow pour historiser le monitoring
    mlflow.set_experiment("data-drift-monitoring")
    with mlflow.start_run(run_name="drift-check"):
        for col, res in results.items():
            mlflow.log_metric(f"psi_{col}", res["psi"])
        mlflow.log_metric("drift_detected", int(drift_detected))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if drift_detected:
        print("DERIVE DETECTEE -> re-entrainement recommande")
        sys.exit(1)

    print("Pas de derive significative")
    sys.exit(0)


if __name__ == "__main__":
    main()
