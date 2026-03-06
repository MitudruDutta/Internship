from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_DIR / "dataset" / "survey_results_feature_engineered.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "artifacts" / "lgbm_price_model.pkl"

RAW_INPUT_FIELDS = [
    "age",
    "gender",
    "zone",
    "occupation",
    "income_levels",
    "consume_frequency(weekly)",
    "current_brand",
    "preferable_consumption_size",
    "awareness_of_other_brands",
    "reasons_for_choosing_brands",
    "flavor_preference",
    "purchase_channel",
    "packaging_preference",
    "health_concerns",
    "typical_consumption_situations",
]

UI_OPTIONS = {
    "gender": ["M", "F"],
    "zone": ["Urban", "Metro", "Semi-Urban", "Rural"],
    "occupation": ["Working Professional", "Student", "Entrepreneur", "Retired"],
    "income_levels": ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L", "Not Reported"],
    "consume_frequency(weekly)": ["0-2 times", "3-4 times", "5-7 times"],
    "current_brand": ["Newcomer", "Established"],
    "preferable_consumption_size": ["Small (250 ml)", "Medium (500 ml)", "Large (1 L)"],
    "awareness_of_other_brands": ["0 to 1", "2 to 4", "above 4"],
    "reasons_for_choosing_brands": ["Price", "Quality", "Availability", "Brand Reputation"],
    "flavor_preference": ["Traditional", "Exotic"],
    "purchase_channel": ["Online", "Retail Store"],
    "packaging_preference": ["Simple", "Premium", "Eco-Friendly"],
    "health_concerns": [
        "Low (Not very concerned)",
        "Medium (Moderately health-conscious)",
        "High (Very health-conscious)",
    ],
    "typical_consumption_situations": [
        "Active (eg. Sports, gym)",
        "Casual (eg. At home)",
        "Social (eg. Parties)",
    ],
}

AGE_GROUP_MAP = {"18-25": 1, "26-35": 2, "36-45": 3, "46-55": 4, "56-70": 5, "70+": 6}
INCOME_LEVELS_MAP = {"Not Reported": 0, "<10L": 1, "10L - 15L": 2, "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5}
HEALTH_CONCERNS_MAP = {
    "Low (Not very concerned)": 1,
    "Medium (Moderately health-conscious)": 2,
    "High (Very health-conscious)": 3,
}
CONSUME_FREQ_MAP = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
AWARENESS_MAP = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
SIZE_MAP = {"Small (250 ml)": 1, "Medium (500 ml)": 2, "Large (1 L)": 3}
ZONE_MAP = {"Rural": 1, "Semi-Urban": 2, "Urban": 3, "Metro": 4}

MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "random_state": 42,
    "objective": "multiclass",
    "n_jobs": 1,
    "verbose": -1,
}


def age_to_group(age: int) -> str:
    if age <= 25:
        return "18-25"
    if age <= 35:
        return "26-35"
    if age <= 45:
        return "36-45"
    if age <= 55:
        return "46-55"
    if age <= 70:
        return "56-70"
    return "70+"


def _prepare_model_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    prepared = features_df.copy()
    prepared["age_group"] = prepared["age_group"].map(AGE_GROUP_MAP)
    prepared["income_levels"] = prepared["income_levels"].map(INCOME_LEVELS_MAP)
    prepared["health_concerns"] = prepared["health_concerns"].map(HEALTH_CONCERNS_MAP)
    prepared["consume_frequency(weekly)"] = prepared["consume_frequency(weekly)"].map(CONSUME_FREQ_MAP)
    prepared["awareness_of_other_brands"] = prepared["awareness_of_other_brands"].map(AWARENESS_MAP)
    prepared["preferable_consumption_size"] = prepared["preferable_consumption_size"].map(SIZE_MAP)
    prepared["zone"] = prepared["zone"].map(ZONE_MAP)

    categorical_cols = prepared.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(prepared, columns=categorical_cols, drop_first=False)
    encoded.columns = encoded.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True).str.strip("_")
    return encoded


def build_engineered_record(raw_input: dict[str, Any]) -> dict[str, Any]:
    missing_fields = [field for field in RAW_INPUT_FIELDS if field not in raw_input]
    if missing_fields:
        raise ValueError(f"Missing input fields: {missing_fields}")

    age = int(raw_input["age"])
    if age < 18:
        raise ValueError("age must be >= 18")

    consume_value = str(raw_input["consume_frequency(weekly)"])
    awareness_value = str(raw_input["awareness_of_other_brands"])
    zone_value = str(raw_input["zone"])
    income_value = str(raw_input["income_levels"])
    brand_value = str(raw_input["current_brand"])
    reason_value = str(raw_input["reasons_for_choosing_brands"])

    cf_score = round(
        CONSUME_FREQ_MAP[consume_value] / (CONSUME_FREQ_MAP[consume_value] + AWARENESS_MAP[awareness_value]),
        2,
    )
    zas_score = int(ZONE_MAP[zone_value] * INCOME_LEVELS_MAP[income_value])
    bsi = int((brand_value != "Established") and (reason_value in {"Price", "Quality"}))

    return {
        "gender": str(raw_input["gender"]),
        "zone": zone_value,
        "occupation": str(raw_input["occupation"]),
        "income_levels": income_value,
        "consume_frequency(weekly)": consume_value,
        "current_brand": brand_value,
        "preferable_consumption_size": str(raw_input["preferable_consumption_size"]),
        "awareness_of_other_brands": awareness_value,
        "reasons_for_choosing_brands": reason_value,
        "flavor_preference": str(raw_input["flavor_preference"]),
        "purchase_channel": str(raw_input["purchase_channel"]),
        "packaging_preference": str(raw_input["packaging_preference"]),
        "health_concerns": str(raw_input["health_concerns"]),
        "typical_consumption_situations": str(raw_input["typical_consumption_situations"]),
        "age_group": age_to_group(age),
        "cf_ab_score": cf_score,
        "zas_score": zas_score,
        "bsi": bsi,
    }


def train_and_save_best_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    source_df = pd.read_csv(data_path)
    X = source_df.drop(columns=["respondent_id", "price_range"]).copy()
    y = source_df["price_range"].copy()

    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y.astype(str))
    X_encoded = _prepare_model_matrix(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y_encoded,
        test_size=0.25,
        random_state=random_state,
        stratify=y_encoded,
    )

    eval_model = LGBMClassifier(**MODEL_PARAMS)
    eval_model.fit(X_train, y_train)
    holdout_accuracy = float(accuracy_score(y_test, eval_model.predict(X_test)))

    final_model = LGBMClassifier(**MODEL_PARAMS)
    final_model.fit(X_encoded, y_encoded)

    artifacts = {
        "model": final_model,
        "feature_columns": X_encoded.columns.tolist(),
        "target_classes": y_encoder.classes_.tolist(),
        "holdout_accuracy": holdout_accuracy,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, model_path)
    return artifacts


def load_model_artifacts(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(model_path)


def ensure_model_artifacts(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_path: str | Path = DEFAULT_DATA_PATH,
    force_retrain: bool = False,
) -> dict[str, Any]:
    model_path = Path(model_path)
    if force_retrain or not model_path.exists():
        return train_and_save_best_model(data_path=data_path, model_path=model_path)
    return load_model_artifacts(model_path=model_path)


def predict_price_range(
    raw_input: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    engineered_record = build_engineered_record(raw_input)
    inference_df = pd.DataFrame([engineered_record])
    encoded_df = _prepare_model_matrix(inference_df)
    aligned_df = encoded_df.reindex(columns=artifacts["feature_columns"], fill_value=0)

    model = artifacts["model"]
    class_labels = list(artifacts["target_classes"])
    probabilities = model.predict_proba(aligned_df)[0]
    predicted_index = int(probabilities.argmax())
    predicted_label = class_labels[predicted_index]

    class_probabilities = {
        label: float(probabilities[index]) for index, label in enumerate(class_labels)
    }
    return {
        "predicted_price_range": predicted_label,
        "class_probabilities": class_probabilities,
        "confidence": float(probabilities[predicted_index]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save the CodeX LightGBM pricing model.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to engineered CSV dataset.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Output path for serialized model.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain and overwrite model artifact.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = ensure_model_artifacts(
        data_path=args.data_path,
        model_path=args.model_path,
        force_retrain=args.force_retrain,
    )
    print(f"Model artifact ready: {Path(args.model_path).resolve()}")
    print(f"Holdout accuracy: {artifacts['holdout_accuracy']:.4f}")
    print(f"Target classes: {artifacts['target_classes']}")
    print(f"Encoded feature count: {len(artifacts['feature_columns'])}")


if __name__ == "__main__":
    main()
