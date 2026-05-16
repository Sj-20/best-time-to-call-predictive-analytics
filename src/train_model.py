# -*- coding: utf-8 -*-
"""
Best Time To Call predictive analytics pipeline.

This script keeps the structure and business logic of the original BTC descriptive
analytics work, then adds a production-friendly ML layer:
1. Load or generate synthetic outbound-call data.
2. Clean and engineer date/time features.
3. Recreate descriptive connect-rate summaries by hour, weekday, month, and time class.
4. Train a predictive model to estimate the probability that a scheduled outbound call connects.
5. Save model artifacts, metrics, feature importance, prediction samples, and summary tables.

Run in Google Colab or locally:
    python src/train_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from generate_synthetic_data import generate_synthetic_btc_data
except ImportError:
    from src.generate_synthetic_data import generate_synthetic_btc_data


PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_btc_outbound_calls.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def time_class(hour: int) -> str:
    if 8 <= hour <= 10:
        return "Morning"
    if 11 <= hour <= 13:
        return "Noon"
    if 14 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 19:
        return "Evening"
    return "Outside Business Hours"


def actual_connect(total_billsec: int, disposition: str, agent_name: str) -> str:
    """Replicates the original business rule used in the descriptive BTC project."""
    if total_billsec >= 20 and disposition == "ANSWERED" and "t1.tlp.tso.ms" in str(agent_name):
        return "Actual Connected"
    return "Not Connected"


def load_or_create_data(data_path: Path = DATA_PATH, rows: int = 120_000) -> pd.DataFrame:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        df = generate_synthetic_btc_data(n_rows=rows)
        df.to_csv(data_path, index=False)
    return pd.read_csv(data_path)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Original Call Date"] = pd.to_datetime(df["Original Call Date"])
    df = df[df["Original Call Hour"].between(8, 19)].copy()

    df["WeekName"] = df["Original Call Date"].dt.day_name()
    df["MonthName"] = df["Original Call Date"].dt.month_name()
    df["Month"] = df["Original Call Date"].dt.month
    df["DayOfWeek"] = df["Original Call Date"].dt.weekday
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["TimeClass"] = df["Original Call Hour"].apply(time_class)
    df["Actual Dispositions"] = df.apply(
        lambda x: actual_connect(x["Total Billsec"], x["Disposition"], x["Agent Name"]), axis=1
    )
    df["ConnectedFlag"] = (df["Actual Dispositions"] == "Actual Connected").astype(int)

    # Historical/call-planning feature: attempt sequence per caller, known before each subsequent attempt.
    df = df.sort_values(["Callerid", "Original Call Date", "Original Call Hour", "Original Call Minute", "Uniqueid"])
    df["CallerAttemptNumber"] = df.groupby("Callerid").cumcount() + 1
    df["CallerAttemptBucket"] = pd.cut(
        df["CallerAttemptNumber"], bins=[0, 1, 2, 3, 5, 999], labels=["1", "2", "3", "4-5", "6+"]
    ).astype(str)
    return df.reset_index(drop=True)


def connect_rate_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(Total_Attempts=("Callerid", "nunique"), Actual_Connected=("ConnectedFlag", "sum"))
        .reset_index()
    )
    grouped["Connect_Rate"] = grouped["Actual_Connected"] / grouped["Total_Attempts"].replace(0, np.nan)
    return grouped.sort_values("Connect_Rate", ascending=False)


def build_model() -> Pipeline:
    numeric_features = ["Original Call Hour", "Original Call Minute", "Month", "DayOfWeek", "IsWeekend", "CallerAttemptNumber"]
    categorical_features = ["WeekName", "MonthName", "TimeClass", "Agent Name", "CallerAttemptBucket"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", classifier)])


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    # thresholds has one fewer element than precision/recall.
    if len(thresholds) == 0:
        return 0.50
    return float(thresholds[np.nanargmax(f1_scores[:-1])])


def evaluate_model(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> Dict:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "f1": round(float(f1_score(y_true, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(y_true, predictions, output_dict=True),
    }
    return metrics


def make_candidate_schedule(model: Pipeline) -> pd.DataFrame:
    """Score candidate time windows for a typical future campaign week."""
    future_dates = pd.date_range("2026-01-05", periods=7, freq="D")
    rows = []
    for date in future_dates:
        for hour in range(8, 20):
            for minute in [0, 15, 30, 45]:
                rows.append(
                    {
                        "Original Call Date": date,
                        "Original Call Hour": hour,
                        "Original Call Minute": minute,
                        "Month": date.month,
                        "DayOfWeek": date.weekday(),
                        "IsWeekend": int(date.weekday() >= 5),
                        "WeekName": date.day_name(),
                        "MonthName": date.month_name(),
                        "TimeClass": time_class(hour),
                        "Agent Name": "t1.tlp.tso.ms.agent_01",
                        "CallerAttemptNumber": 1,
                        "CallerAttemptBucket": "1",
                    }
                )
    candidates = pd.DataFrame(rows)
    candidates["Predicted_Connect_Probability"] = model.predict_proba(candidates)[:, 1]
    return candidates.sort_values("Predicted_Connect_Probability", ascending=False)


def save_feature_importance(model: Pipeline, output_path: Path) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "importance": rf.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    importance.to_csv(output_path, index=False)
    return importance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_or_create_data(data_path, rows=args.rows)
    df = prepare_dataset(raw_df)

    # Descriptive analytics outputs similar to the original script, but less repetitive and GitHub-ready.
    summary_specs = {
        "connect_rate_by_hour.csv": ["Original Call Hour"],
        "connect_rate_by_weekday.csv": ["WeekName"],
        "connect_rate_by_month.csv": ["MonthName"],
        "connect_rate_by_timeclass.csv": ["TimeClass"],
        "connect_rate_by_weekday_hour.csv": ["WeekName", "Original Call Hour"],
        "connect_rate_by_agent_timeclass.csv": ["Agent Name", "TimeClass"],
    }
    for filename, cols in summary_specs.items():
        connect_rate_summary(df, cols).to_csv(output_dir / filename, index=False)

    feature_cols = [
        "Original Call Hour",
        "Original Call Minute",
        "Month",
        "DayOfWeek",
        "IsWeekend",
        "WeekName",
        "MonthName",
        "TimeClass",
        "Agent Name",
        "CallerAttemptNumber",
        "CallerAttemptBucket",
    ]
    target_col = "ConnectedFlag"

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, test_prob)
    metrics = evaluate_model(y_test, test_prob, threshold)

    joblib.dump(model, model_dir / "btc_connect_probability_model.joblib")
    with open(model_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_feature_importance(model, output_dir / "feature_importance.csv")

    scored_test = X_test.copy()
    scored_test["Actual_Connected"] = y_test.values
    scored_test["Predicted_Connect_Probability"] = test_prob
    scored_test["Predicted_Label"] = (test_prob >= threshold).astype(int)
    scored_test.to_csv(output_dir / "test_predictions_sample.csv", index=False)

    candidates = make_candidate_schedule(model)
    candidates.to_csv(output_dir / "future_best_time_recommendations.csv", index=False)
    candidates.groupby(["WeekName", "Original Call Hour"], as_index=False)["Predicted_Connect_Probability"].mean() \
        .sort_values("Predicted_Connect_Probability", ascending=False) \
        .to_csv(output_dir / "best_hour_by_weekday_recommendations.csv", index=False)

    print("BTC predictive analytics pipeline completed.")
    print(f"Rows used: {len(df):,}")
    print(f"Connection rate: {df['ConnectedFlag'].mean():.2%}")
    print(f"Model ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']} | F1: {metrics['f1']}")
    print(f"Artifacts saved to: {model_dir.resolve()}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
