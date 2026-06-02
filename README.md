# Data Science & Machine Learning — Project Portfolio

End-to-end data science work across three business problems — from FMCG promotion analytics and statistical/SQL analysis to a fully deployed machine-learning product for beverage price prediction.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
<img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-92.8%25-9ACD32">
<img alt="MLflow" src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white">
<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white">
</p>

**Live experiment tracking:** [MLflow on DagsHub](https://dagshub.com/MitudruDutta/CodeX_Project.mlflow)

---

## ⭐ Highlight Project — CodeX: Beverage Price Prediction

An end-to-end ML system that predicts the **price band a consumer will accept** for a new energy drink, from a 30,000-person market survey — and serves it as a live web app.

> **Result: LightGBM at 92.8% test accuracy, balanced (F1 0.91–0.95) across all four price bands.**

### The pipeline

```
Raw survey (30,010)  →  Cleaning (29,991)  →  Feature engineering (29,956)
        →  6-model comparison  →  MLflow + DagsHub tracking  →  FastAPI + Streamlit deployment
```

### Model comparison

| Model | Test Accuracy |
|-------|:-------------:|
| **LightGBM** ⭐ | **0.9278** |
| XGBoost | 0.9230 |
| Random Forest | 0.8989 |
| Support Vector Machine | 0.8463 |
| Logistic Regression | 0.8348 |
| Gaussian Naive Bayes | 0.5877 |

### What it does

- **Cleaning** — removed duplicates, capped impossible ages (max raw age was 604), treated 8,060 missing incomes as a `Not Reported` category, standardized typos (`Metor → Metro`).
- **Feature engineering** — 4 engineered signals: `age_group`, `cf_ab_score` (loyalty), `zas_score` (Zone Affluence Score), `bsi` (Brand Switching Indicator).
- **Modeling** — 32 encoded features, stratified 75/25 split, 6 classifiers compared on identical data.
- **Tracking** — every run logged to MLflow on DagsHub (params, metrics, classification reports, serialized models) — fully reproducible.
- **Deployment** — Streamlit form → FastAPI `/predict` → LightGBM, returning the price band, confidence, and full class probabilities.

### Run CodeX locally

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd "Week 3 and 4/CodeX Project"

# (re)train the serialized model artifact if needed
python backend/model_helper.py --force-retrain

# start the API
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000

# in a second terminal: start the app
streamlit run app.py
```

---

## Repository structure

```
Data-Science-Internship/
├── Week 1/                     # Nova Mart FMCG promotion analysis
│   ├── Task 1/task1.ipynb      #   client-request answers (IR%, ISU%, coverage)
│   └── Task 2/task2.ipynb      #   presentation-ready charts
├── Week 2/                     # Statistics + SQL debugging
│   ├── Task 1/task1.ipynb      #   descriptive & inferential stats
│   └── Task 2/sql/...          #   fixed EV-market SQL queries
├── Week 3 and 4/
│   └── CodeX Project/          # ⭐ end-to-end ML pricing product
│       ├── datacleaning.ipynb
│       ├── featureengineering.ipynb
│       ├── model.ipynb
│       ├── mlflow_tracking.py
│       ├── backend/            # FastAPI service + model helper
│       ├── app.py              # Streamlit front end
│       └── artifacts/          # serialized LightGBM model
├── requirements.txt
└── README.md
```

## Projects at a glance

| Week | Project | Focus | Stack |
|------|---------|-------|-------|
| 1 | **Nova Mart** promotion analysis | FMCG campaign lift, `IR%` / `ISU%`, category & store performance | pandas, matplotlib, seaborn |
| 2 | **Statistics + SQL** | Descriptive/inferential stats on e-commerce; debugged EV-market SQL (penetration, CAGR) | pandas, scipy, SQL |
| 3–4 | **CodeX** beverage pricing ⭐ | Full ML lifecycle: clean → feature-engineer → model → track → deploy | scikit-learn, XGBoost, LightGBM, MLflow, FastAPI, Streamlit |

Each week solves a distinct client-style problem — read the repository as a set of project deliverables. Per-week details live in each folder's `README.md`.

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `LightGBM` · `MLflow` · `DagsHub` · `FastAPI` · `Uvicorn` · `Streamlit` · `Matplotlib` · `Seaborn` · `SQL`

## Author

**Mitudru Dutta**
- GitHub: [@MitudruDutta](https://github.com/MitudruDutta)
- Project repo: [github.com/MitudruDutta/Internship](https://github.com/MitudruDutta/Data-Science-Internship)

## Notes

- Some client instruction PDFs and locally generated presentation assets are kept local / out of version control.
- The repo mixes notebook-based analysis (Weeks 1–2) with deployable application code (Week 3–4), so run commands differ by week.
