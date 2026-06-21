
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_REPO = "Greeshmadudu/tourism-dataset"
MODEL_REPO = "Greeshmadudu/tourism-wellness-model"
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "tourism_model.joblib"


def load_split(filename):
    local_path = ARTIFACT_DIR / filename
    if local_path.exists():
        return pd.read_csv(local_path)

    downloaded = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
    )
    return pd.read_csv(downloaded)


def build_pipeline(train_df):
    X = train_df.drop(columns=["ProdTaken"])
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    train_df = load_split("train.csv")
    test_df = load_split("test.csv")

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]

    mlflow.set_experiment("tourism-wellness-package")
    with mlflow.start_run():
        pipeline = build_pipeline(train_df)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_metrics(metrics)
        mlflow.log_text(classification_report(y_test, predictions), "classification_report.txt")
        mlflow.sklearn.log_model(pipeline, "model")

        joblib.dump(pipeline, MODEL_PATH)
        upload_file(
            path_or_fileobj=str(MODEL_PATH),
            path_in_repo="tourism_model.joblib",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(metrics)


if __name__ == "__main__":
    main()
