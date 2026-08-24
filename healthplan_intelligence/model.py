"""Train and use the project's intentionally small machine-learning model."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURE_COLUMNS


NUMERIC_FEATURES = [
    "deductible",
    "primary_care_copay",
    "coinsurance_percent",
    "annual_premium",
]
CATEGORICAL_FEATURES = ["plan_type", "hsa_eligible"]


def build_model(seed: int = 42) -> Pipeline:
    """Build a readable preprocessing-and-classification pipeline."""

    preparation = ColumnTransformer(
        transformers=[
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=160,
        max_depth=7,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("prepare_inputs", preparation),
            ("estimate_risk", classifier),
        ]
    )


def train_model(
    data: pd.DataFrame, seed: int = 42
) -> tuple[Pipeline, dict[str, float], dict[str, float]]:
    """Train the model and return model, test metrics, and human benchmarks."""

    missing = set(FEATURE_COLUMNS + ["high_out_of_pocket"]) - set(data.columns)
    if missing:
        raise ValueError(f"The training data is missing: {sorted(missing)}")

    inputs = data[FEATURE_COLUMNS]
    target = data["high_out_of_pocket"]
    train_inputs, test_inputs, train_target, test_target = train_test_split(
        inputs,
        target,
        test_size=0.25,
        random_state=seed,
        stratify=target,
    )

    model = build_model(seed=seed)
    model.fit(train_inputs, train_target)

    predictions = model.predict(test_inputs)
    probabilities = model.predict_proba(test_inputs)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(test_target, predictions)),
        "roc_auc": float(roc_auc_score(test_target, probabilities)),
    }
    benchmarks = {
        column: float(data[column].median()) for column in NUMERIC_FEATURES
    }
    return model, metrics, benchmarks


def predict_high_cost_risk(model: Pipeline, plan: dict[str, Any]) -> float:
    """Return the model's high-cost probability for one health plan."""

    missing = set(FEATURE_COLUMNS) - set(plan)
    if missing:
        raise ValueError(f"The plan is missing: {sorted(missing)}")

    plan_frame = pd.DataFrame([{name: plan[name] for name in FEATURE_COLUMNS}])
    return float(model.predict_proba(plan_frame)[0, 1])
