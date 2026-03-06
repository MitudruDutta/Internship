# MLflow + DagsHub Tracking (CodeX Project)

Run from:

`Week 3 and 4/CodeX Project`

## 1) Activate environment

```bash
source ~/python/bin/activate
```

## 2) Install dependencies (if missing)

```bash
pip install mlflow scikit-learn xgboost lightgbm
```

## 3) Local MLflow run (quick validation)

```bash
python mlflow_tracking.py
```

This logs all six models into your configured MLflow backend (default: local if no tracking URI is set).

## 4) Publish runs to DagsHub MLflow

Set your DagsHub credentials first:

```bash
export DAGSHUB_OWNER="<your_dagshub_username_or_org>"
export DAGSHUB_REPO="<your_repo_name>"
export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
export MLFLOW_TRACKING_PASSWORD="<your_dagshub_access_token>"
```

Then run:

```bash
python mlflow_tracking.py --dagshub-owner "$DAGSHUB_OWNER" --dagshub-repo "$DAGSHUB_REPO"
```

The script prints:
- model comparison summary
- best model
- MLflow shareable link in this format:
  - `https://dagshub.com/<owner>/<repo>.mlflow/#/experiments/<experiment_id>`

## 5) Optional direct tracking URI usage

```bash
python mlflow_tracking.py --tracking-uri "https://dagshub.com/<owner>/<repo>.mlflow"
```
