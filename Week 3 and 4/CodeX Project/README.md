# CodeX Energy Drink Pricing Prediction

This project is a comprehensive Machine Learning solution developed to predict the optimal price range for energy drinks based on consumer survey responses. It encompasses the entire data lifecycle: from data cleaning and feature engineering to predictive modeling and deployment via an interactive web application.

## Project Structure

- **`dataset/`**: Contains the raw, cleaned, and feature-engineered survey data (ignored in version control).
- **`datacleaning.ipynb`**: Notebook detailing the procedures for handling missing values, outliers, and formatting the raw survey data.
- **`featureengineering.ipynb`**: Notebook focusing on creating new derived features (like `cf_ab_score`, `zas_score`, and `bsi`) to improve model performance.
- **`model.ipynb`**: Notebook exploring various classification models (LightGBM, XGBoost, Random Forest, etc.) to determine the best approach for price prediction.
- **`mlflow_tracking.py`**: Script for logging model parameters, metrics, and artifacts using MLflow for experiment tracking.
- **`backend/`**:
  - **`fastapi_app.py`**: A FastAPI application that serves the best performing LightGBM model as a REST API endpoint (`/predict`).
  - **`model_helper.py`**: A helper module that preprocesses incoming raw data using the same mappings defined during feature engineering and runs inference using the serialized model.
- **`app.py`**: An interactive Streamlit frontend that provides a user-friendly form to input consumer demographics, habits, and preferences, and displays the predicted price range along with class probabilities.
- **`artifacts/`**: Directory containing the serialized `.pkl` LightGBM model utilized by the backend.

## Setup and Installation

1. Ensure you have Python installed and your virtual environment activated.
2. Install the necessary dependencies (refer to the main repository `requirements.txt`).
3. If you need to retrain the model, you can run:
   ```bash
   python backend/model_helper.py --force-retrain
   ```

## Running the Application

The application is split into a FastAPI backend and a Streamlit frontend. They should be run in separate terminal sessions.

### 1. Start the FastAPI Backend
```bash
# From the CodeX Project directory
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

### 2. Start the Streamlit Frontend
```bash
# From the CodeX Project directory
streamlit run app.py
```
The web interface will open in your default browser at `http://localhost:8501`.

## Features
- Dynamic form matching the features required by the predictive model.
- Immediate prediction of the optimal price range (`50-100`, `100-150`, `150-200`, `200-250`).
- Visual bar chart representing the confidence probabilities across all possible price ranges.

## Instructions & Demos
*Note: Some instruction PDFs and demo GIFs are ignored from version control to save space.*
