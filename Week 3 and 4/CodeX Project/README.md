# CodeX Beverage Price Prediction

This project predicts the expected price range of an energy drink from consumer survey responses. It covers the full workflow from raw survey cleanup to model deployment.

## Problem statement

The business goal is to estimate which price band a respondent is most likely to accept. The target variable is `price_range`, and the current implementation treats this as a multi-class classification problem.

The survey captures demographics, purchase behavior, brand awareness, packaging preference, health concerns, and consumption context. Those attributes are cleaned, transformed, and then used to predict one of these ranges:

- `50-100`
- `100-150`
- `150-200`
- `200-250`

## Workflow

### 1. Data cleaning

`datacleaning.ipynb` prepares the survey data by handling nulls, category standardization, and data quality issues based on the instructions and metadata.

### 2. Feature engineering

`featureengineering.ipynb` creates the derived fields used in modeling:

- `age_group`
- `cf_ab_score`
- `zas_score`
- `bsi`

The notebook also removes logical outliers before exporting the model-ready dataset.

### 3. Predictive modeling

`model.ipynb` compares multiple classifiers, including LightGBM, XGBoost, Random Forest, SVM, Logistic Regression, and Gaussian Naive Bayes.

Based on the saved comparison results in `dataset/mlflow_model_comparison_results.csv`, the best-performing model is `LGBMClassifier` with accuracy of about `0.9278`.

### 4. MLflow tracking

`mlflow_tracking.py` logs model parameters, metrics, reports, and the best-model summary. It supports local MLflow tracking and DagsHub-compatible tracking URIs.

### 5. Deployment

The deployed setup has three components:

- `backend/model_helper.py`: builds engineered features from raw request payloads, loads the serialized model, and returns prediction probabilities
- `backend/fastapi_app.py`: exposes the prediction API
- `app.py`: Streamlit front end for entering respondent details and showing predicted price range plus class probabilities

The current serialized model artifact is:

- `artifacts/lgbm_price_model.pkl`

## Key files

- `dataset/survey_results.csv`
- `dataset/survey_results_cleaned.csv`
- `dataset/survey_results_feature_engineered.csv`
- `datacleaning.ipynb`
- `featureengineering.ipynb`
- `model.ipynb`
- `mlflow_tracking.py`
- `backend/model_helper.py`
- `backend/fastapi_app.py`
- `app.py`

## How to run

From the repository root, install dependencies:

```bash
pip install -r requirements.txt
```

From `Week 3 and 4/CodeX Project`, retrain or regenerate the model artifact if needed:

```bash
python backend/model_helper.py --force-retrain
```

Start the FastAPI server:

```bash
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000
```

Start the Streamlit app in a second terminal:

```bash
streamlit run app.py
```

## Notes

- `streamlit_app_demo.gif` and `ml_flow_output_demo.png` are local reference assets for the app and MLflow output.
- The API expects the same raw survey fields used in the Streamlit form, and the helper handles engineered features internally before inference.
