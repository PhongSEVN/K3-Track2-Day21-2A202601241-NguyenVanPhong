import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70
CLASS_NAMES = {0: "thap", 1: "trung_binh", 2: "cao"}
DRIFT_MIN_SHARE = 0.10
METRICS_KEY = "models/latest/metrics.json"

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def build_model(model_type: str, hp: dict):
    """Bonus 2: chon thuat toan theo model_type trong params.yaml."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}")
    return MODEL_REGISTRY[model_type](**hp, random_state=42)


def label_distribution(y: pd.Series) -> dict:
    counts = y.value_counts(normalize=True).sort_index()
    return {str(k): float(v) for k, v in counts.items()}


def check_drift(distribution: dict) -> list[str]:
    """Bonus 5: canh bao lop nao chiem duoi DRIFT_MIN_SHARE tong mau."""
    warnings = []
    for label, share in distribution.items():
        if share < DRIFT_MIN_SHARE:
            warnings.append(
                f"Class {label} chi chiem {share:.1%} tong mau (< {DRIFT_MIN_SHARE:.0%})"
            )
    return warnings


def write_report(y_eval, preds) -> None:
    """Bonus 3: confusion matrix + precision/recall tung lop, ghi outputs/report.txt."""
    cm = confusion_matrix(y_eval, preds, labels=sorted(CLASS_NAMES))
    precision, recall, _, _ = precision_recall_fscore_support(
        y_eval, preds, labels=sorted(CLASS_NAMES), zero_division=0
    )

    lines = ["Confusion matrix (rows=true, cols=pred):", str(cm), ""]
    lines.append(f"{'class':<12}{'precision':>10}{'recall':>10}")
    for idx, label in enumerate(sorted(CLASS_NAMES)):
        lines.append(f"{CLASS_NAMES[label]:<12}{precision[idx]:>10.4f}{recall[idx]:>10.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


def fetch_previous_accuracy(bucket_name: str | None) -> float | None:
    """Bonus 4: doc accuracy cua model dang deploy tren GCS de lam rollback gate."""
    if not bucket_name:
        return None
    try:
        from google.cloud import storage

        client = storage.Client()
        blob = client.bucket(bucket_name).blob(METRICS_KEY)
        if not blob.exists():
            return None
        data = json.loads(blob.download_as_text())
        return data.get("accuracy")
    except Exception as exc:  # noqa: BLE001 - rollback gate khong duoc lam crash training
        print(f"[WARN] Khong doc duoc metrics cu tu GCS: {exc}")
        return None


def should_deploy(new_accuracy: float, previous_accuracy: float | None) -> bool:
    """Bonus 4: chi cho phep deploy neu model moi khong te hon model dang chay."""
    if previous_accuracy is None:
        return True
    return new_accuracy >= previous_accuracy


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua "model_type" va sieu tham so tuong ung cho thuat toan do.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    model_type = params.get("model_type", "random_forest")
    hp = params.get(model_type, {})

    distribution = label_distribution(y_train)
    drift_warnings = check_drift(distribution)
    for warning in drift_warnings:
        print(f"[DRIFT WARNING] {warning}")

    bucket_name = os.environ.get("GCS_BUCKET")
    previous_accuracy = fetch_previous_accuracy(bucket_name)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(hp)

        model = build_model(model_type, hp)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        deploy_ok = bool(should_deploy(acc, previous_accuracy))
        mlflow.log_param("deploy_ok", deploy_ok)

        print(f"[{model_type}] Accuracy: {acc:.4f} | F1: {f1:.4f} | deploy_ok={deploy_ok}")
        if not deploy_ok:
            print(f"[ROLLBACK GATE] {acc:.4f} < previous {previous_accuracy:.4f}, khong deploy.")

        write_report(y_eval, preds)

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "model_type": model_type,
                    "label_distribution": distribution,
                    "drift_warnings": drift_warnings,
                    "deploy_ok": deploy_ok,
                },
                f,
                indent=2,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
