from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(slots=True)
class ModelResult:
    cohort: str
    model_name: str
    metrics: dict[str, float | str]
    predictions: pd.DataFrame


def _binary_log_loss(y_true: pd.Series | np.ndarray, probability: pd.Series | np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1]))


def _classification_metrics(y_true: pd.Series | np.ndarray, probability: pd.Series | np.ndarray) -> dict[str, float]:
    y_actual = pd.Series(y_true).astype(int)
    y_probability = pd.Series(probability).clip(1e-6, 1 - 1e-6).astype(float)
    y_pred = (y_probability >= 0.5).astype(int)

    metrics = {
        "brier_score": float(brier_score_loss(y_actual, y_probability)),
        "log_loss": _binary_log_loss(y_actual, y_probability),
        "accuracy": float(accuracy_score(y_actual, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_actual, y_pred)),
        "precision": float(precision_score(y_actual, y_pred, zero_division=0)),
        "recall": float(recall_score(y_actual, y_pred, zero_division=0)),
        "f1": float(f1_score(y_actual, y_pred, zero_division=0)),
        "positive_rate": float(y_actual.mean()),
    }
    if y_actual.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_actual, y_probability))
        metrics["average_precision"] = float(average_precision_score(y_actual, y_probability))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def _split_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {"won_match", "cohort", "team_id", "opponent_team_id", "match_id", "source_name"}
    usable = [c for c in df.columns if c not in excluded and not df[c].isna().all()]
    numeric = [c for c in usable if df[c].dtype != "object"]
    categorical = [c for c in usable if df[c].dtype == "object"]
    return numeric, categorical


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _split_features(df)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def train_sklearn_models(match_level: pd.DataFrame) -> list[ModelResult]:
    results: list[ModelResult] = []
    for cohort, cohort_df in match_level.groupby("cohort"):
        cohort_df = cohort_df.sort_values("match_id").reset_index(drop=True)
        split_idx = max(1, int(len(cohort_df) * 0.8))
        train_df = cohort_df.iloc[:split_idx].copy()
        test_df = cohort_df.iloc[split_idx:].copy()
        if test_df.empty:
            test_df = train_df.copy()

        X_train = train_df.drop(columns=["won_match"])
        y_train = train_df["won_match"]
        X_test = test_df.drop(columns=["won_match"])
        y_test = test_df["won_match"]

        preprocessor = _build_preprocessor(X_train)
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=7),
        }
        baseline_prob = 1 / (1 + np.exp(-test_df["opponent_gap_rating_mean"].fillna(0)))
        results.append(
            ModelResult(
                cohort=cohort,
                model_name="static_baseline",
                metrics=_classification_metrics(y_test, baseline_prob),
                predictions=pd.DataFrame({"match_id": test_df["match_id"], "probability": baseline_prob, "actual": y_test}),
            )
        )

        for model_name, estimator in models.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            min_class_count = int(y_train.value_counts().min()) if not y_train.empty else 0
            if len(train_df) >= 6 and min_class_count >= 2:
                calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=min(3, min_class_count))
                calibrated.fit(X_train, y_train)
                probabilities = calibrated.predict_proba(X_test)[:, 1]
            else:
                pipeline.fit(X_train, y_train)
                probabilities = pipeline.predict_proba(X_test)[:, 1]
            results.append(
                ModelResult(
                    cohort=cohort,
                    model_name=model_name,
                    metrics=_classification_metrics(y_test, probabilities),
                    predictions=pd.DataFrame({"match_id": test_df["match_id"], "probability": probabilities, "actual": y_test}),
                )
            )
    return results


def train_torch_benchmark(match_level: pd.DataFrame) -> list[ModelResult]:
    if torch is None:
        return []

    results: list[ModelResult] = []
    for cohort, cohort_df in match_level.groupby("cohort"):
        if len(cohort_df) < 4:
            continue
        numeric_columns = [c for c in cohort_df.columns if c not in {"won_match", "cohort", "team_id", "opponent_team_id", "match_id", "source_name", "map_name", "event_tier"}]
        data = cohort_df[numeric_columns].fillna(0.0).to_numpy(dtype=np.float32)
        target = cohort_df["won_match"].to_numpy(dtype=np.float32)
        split_idx = max(1, int(len(cohort_df) * 0.8))
        X_train, X_test = data[:split_idx], data[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        if len(X_test) == 0:
            X_test, y_test = X_train, y_train

        model = nn.Sequential(nn.Linear(X_train.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)), batch_size=64, shuffle=True)

        model.train()
        for _ in range(25):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_X), batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(torch.from_numpy(X_test))).squeeze(1).numpy()

        results.append(
            ModelResult(
                cohort=cohort,
                model_name="torch_mlp_benchmark",
                metrics=_classification_metrics(y_test, probabilities),
                predictions=pd.DataFrame(
                    {
                        "match_id": cohort_df.iloc[split_idx:]["match_id"].tolist() or cohort_df["match_id"].tolist(),
                        "probability": probabilities,
                        "actual": y_test,
                    }
                ),
            )
        )
    return results
