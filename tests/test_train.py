import os
import json
import numpy as np
import pandas as pd
import pytest
from src.train import (
    build_model,
    check_drift,
    fetch_previous_accuracy,
    label_distribution,
    should_deploy,
    train,
)


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi GCS hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "label_distribution" in metrics
    assert "deploy_ok" in metrics


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_report_file_created(tmp_path):
    """Bonus 3: outputs/report.txt duoc tao voi confusion matrix + precision/recall."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 10, "max_depth": 3}},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")
    with open("outputs/report.txt") as f:
        content = f.read()
    assert "Confusion matrix" in content
    assert "precision" in content
    assert "recall" in content


def test_train_with_gradient_boosting(tmp_path):
    """Bonus 2: train() phai chay duoc voi model_type khac random_forest."""
    train_path, eval_path = _make_temp_data(tmp_path)
    acc = train(
        {
            "model_type": "gradient_boosting",
            "gradient_boosting": {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.5},
        },
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc, float)


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError):
        build_model("unknown", {})


def test_label_distribution_sums_to_one():
    y = pd.Series([0, 0, 1, 1, 2])
    dist = label_distribution(y)
    assert set(dist.keys()) == {"0", "1", "2"}
    assert pytest.approx(sum(dist.values()), abs=1e-9) == 1.0


def test_check_drift_flags_underrepresented_class():
    dist = {"0": 0.46, "1": 0.46, "2": 0.08}
    warnings = check_drift(dist)
    assert len(warnings) == 1
    assert "Class 2" in warnings[0]


def test_check_drift_no_warning_when_balanced():
    dist = {"0": 0.33, "1": 0.34, "2": 0.33}
    assert check_drift(dist) == []


def test_should_deploy_true_when_no_previous():
    assert should_deploy(0.9, None) is True


def test_should_deploy_true_when_better_or_equal():
    assert should_deploy(0.9, 0.85) is True
    assert should_deploy(0.9, 0.9) is True


def test_should_deploy_false_when_regression():
    assert should_deploy(0.8, 0.9) is False


def test_fetch_previous_accuracy_returns_none_without_bucket():
    assert fetch_previous_accuracy(None) is None
