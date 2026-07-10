# Pipeline MLOps - Iris (Azure ML + VS Code)

Pipeline MLOps de bout en bout : entrainement -> tracking MLflow ->
enregistrement -> deploiement endpoint -> A/B testing.

## Developpement (local, VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/train.py --data data/iris.csv --n_estimators 200
mlflow ui                        # http://127.0.0.1:5000
```

## Industrialisation (Azure ML, pilote depuis VS Code)
```bash
az ml job create -f azure-ml/job.yaml -g rg-mlops -w ws-mlops
```
