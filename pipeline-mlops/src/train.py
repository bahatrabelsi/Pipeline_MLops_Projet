import argparse

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Chemin vers le fichier iris.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--target_column", type=str, default="species")
    return parser.parse_args()


def load_data(path: str, target_column: str):
    """Charge le CSV et separe features / cible."""
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(
            f"Colonne cible '{target_column}' absente. "
            f"Colonnes disponibles : {list(df.columns)}"
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def train_model(X_train, y_train, n_estimators: int, max_depth):
    """Entraine un RandomForest et le retourne."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Calcule accuracy et f1 ponderee (multi-classes)."""
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return accuracy, f1


def main():
    args = parse_args()

    # En LOCAL uniquement, decommenter pour viser un serveur MLflow dedie :
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("iris-training-pipeline")

    # autolog gere automatiquement : parametres, metriques d'entrainement
    # ET l'enregistrement du modele au format MLflow. On NE refait donc PAS
    # log_model() a la main (sinon le modele serait enregistre deux fois).
    mlflow.sklearn.autolog()

    X, y = load_data(args.data, args.target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,            # garde les 3 classes equilibrees
    )

    with mlflow.start_run() as run:
        model = train_model(X_train, y_train, args.n_estimators, args.max_depth)
        accuracy, f1 = evaluate(model, X_test, y_test)

        # metriques personnalisees sur le jeu de TEST
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Run ID   : {run.info.run_id}")

    print("Run termine")


if __name__ == "__main__":
    main()
