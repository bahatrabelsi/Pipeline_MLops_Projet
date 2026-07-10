import os
import json
import logging

import pandas as pd
import mlflow.sklearn

model = None

# Ordre des colonnes attendu par le modele (features Iris).
EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def init():
    """Charge le modele MLflow depuis le dossier monte par Azure ML."""
    global model
    model_root = os.getenv("AZUREML_MODEL_DIR", ".")
    # Le modele MLflow est enregistre sous un sous-dossier "model".
    model_path = os.path.join(model_root, "model")
    if not os.path.exists(model_path):
        model_path = model_root  # fallback selon l'enregistrement
    model = mlflow.sklearn.load_model(model_path)
    logging.info("Modele charge depuis %s", model_path)


def run(raw_data):
    """
    Attend un JSON de la forme :
    {
      "data": [
        {"sepal_length": 5.1, "sepal_width": 3.5,
         "petal_length": 1.4, "petal_width": 0.2}
      ]
    }
    """
    try:
        payload = json.loads(raw_data)
        records = payload["data"]
        df = pd.DataFrame(records)

        # securise l'ordre/les noms de colonnes
        df = df[EXPECTED_COLUMNS]

        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}

    except KeyError:
        logging.exception("Cle 'data' manquante dans le payload")
        return {"error": "Le corps doit contenir une cle 'data'."}
    except Exception as exc:  # noqa: BLE001
        logging.exception("Erreur d'inference")
        return {"error": str(exc)}
