import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track CodeX models in MLflow and publish to DagsHub-compatible tracking URI.")
    parser.add_argument(
        "--data-path",
        default="dataset/survey_results_feature_engineered.csv",
        help="Path to engineered dataset.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Beverage Price Prediction",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train_test_split.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="",
        help="Optional MLflow tracking URI override.",
    )
    parser.add_argument(
        "--dagshub-owner",
        default=os.getenv("DAGSHUB_OWNER", ""),
        help="DagsHub owner/user/org (optional).",
    )
    parser.add_argument(
        "--dagshub-repo",
        default=os.getenv("DAGSHUB_REPO", ""),
        help="DagsHub repository name (optional).",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Directory for generated artifacts.",
    )
    return parser.parse_args()


def prepare_features(data_path: str) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    df = pd.read_csv(data_path)

    X = df.drop(columns=["respondent_id", "price_range"]).copy()
    y = df["price_range"].copy()

    age_group_map = {"18-25": 1, "26-35": 2, "36-45": 3, "46-55": 4, "56-70": 5, "70+": 6}
    income_levels_map = {"Not Reported": 0, "<10L": 1, "10L - 15L": 2, "16L - 25L": 3, "26L - 35L": 4, "> 35L": 5}
    health_concerns_map = {
        "Low (Not very concerned)": 1,
        "Medium (Moderately health-conscious)": 2,
        "High (Very health-conscious)": 3,
    }
    consume_freq_map = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
    awareness_map = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}
    size_map = {"Small (250 ml)": 1, "Medium (500 ml)": 2, "Large (1 L)": 3}
    zone_map = {"Rural": 1, "Semi-Urban": 2, "Urban": 3, "Metro": 4}

    X["age_group"] = X["age_group"].map(age_group_map)
    X["income_levels"] = X["income_levels"].map(income_levels_map)
    X["health_concerns"] = X["health_concerns"].map(health_concerns_map)
    X["consume_frequency(weekly)"] = X["consume_frequency(weekly)"].map(consume_freq_map)
    X["awareness_of_other_brands"] = X["awareness_of_other_brands"].map(awareness_map)
    X["preferable_consumption_size"] = X["preferable_consumption_size"].map(size_map)
    X["zone"] = X["zone"].map(zone_map)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X_encoded.columns = X_encoded.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True).str.strip("_")

    target_encoder = LabelEncoder()
    y_encoded = pd.Series(target_encoder.fit_transform(y.astype(str)), name="price_range")
    return X_encoded, y_encoded, target_encoder


def build_models(random_state: int) -> dict[str, object]:
    return {
        "LGBMClassifier": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=random_state,
            objective="multiclass",
            n_jobs=1,
            verbose=-1,
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=1,
            verbosity=0,
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1),
        "Support Vector": SVC(kernel="rbf", C=1.0, gamma="scale"),
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "Gaussian Naive Bayes": GaussianNB(),
    }


def configure_tracking(args: argparse.Namespace) -> tuple[str, str]:
    tracking_uri = args.tracking_uri.strip()
    if not tracking_uri and args.dagshub_owner and args.dagshub_repo:
        tracking_uri = f"https://dagshub.com/{args.dagshub_owner}/{args.dagshub_repo}.mlflow"

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.experiment_name)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    experiment_id = experiment.experiment_id if experiment else ""
    return tracking_uri, experiment_id


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri, experiment_id = configure_tracking(args)

    X, y, target_encoder = prepare_features(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = build_models(args.random_state)
    results: list[dict] = []

    for run_name, model in models.items():
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", run_name)
            mlflow.log_param("rows_total", int(X.shape[0]))
            mlflow.log_param("features_total", int(X.shape[1]))
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("target_classes", list(target_encoder.classes_))

            if hasattr(model, "get_params"):
                for key, value in model.get_params().items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        mlflow.log_param(f"model__{key}", value)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("f1_weighted", float(weighted_f1))
            mlflow.log_metric("f1_macro", float(macro_f1))

            report_dict = classification_report(
                y_test,
                y_pred,
                target_names=target_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
            report_path = output_dir / f"classification_report_{run_name.replace(' ', '_')}.json"
            report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path))

            mlflow.sklearn.log_model(model, artifact_path="model")

            results.append(
                {
                    "Model": run_name,
                    "Accuracy": float(accuracy),
                    "F1_weighted": float(weighted_f1),
                    "F1_macro": float(macro_f1),
                }
            )

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    results_path = output_dir / "mlflow_model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    best_model = results_df.loc[0, "Model"]
    best_acc = float(results_df.loc[0, "Accuracy"])

    with mlflow.start_run(run_name="Model_Comparison_Summary"):
        mlflow.log_artifact(str(results_path))
        mlflow.log_metric("best_accuracy", best_acc)
        mlflow.log_param("best_model", best_model)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model} ({best_acc:.4f})")
    print(f"Results saved: {results_path}")

    if tracking_uri.startswith("https://dagshub.com/") and experiment_id:
        share_link = f"{tracking_uri}/#/experiments/{experiment_id}"
        print(f"\nMLflow shareable link: {share_link}")
    elif experiment_id:
        print(f"\nMLflow experiment id: {experiment_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
