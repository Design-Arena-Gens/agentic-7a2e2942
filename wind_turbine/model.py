from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .data import derive_feature_targets


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_names: List[str]
    train_report: Dict[str, Dict[str, float]]
    test_report: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    threshold: float


def build_training_pipeline() -> Pipeline:
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", RobustScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numerical_transformer, slice(0, None))]
    )
    classifier = RandomForestClassifier(
        n_estimators=350, max_depth=9, min_samples_leaf=3, random_state=42
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", classifier)])
    return pipeline


def train_failure_model(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ModelArtifacts:
    X, y, feature_names = derive_feature_targets(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = build_training_pipeline()
    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict_proba(X_train)[:, 1]
    test_preds = pipeline.predict_proba(X_test)[:, 1]

    train_report = classification_report(
        y_train, (train_preds >= threshold).astype(int), output_dict=True
    )
    test_report = classification_report(
        y_test, (test_preds >= threshold).astype(int), output_dict=True
    )

    cm = confusion_matrix(
        y_test, (test_preds >= threshold).astype(int), labels=[0, 1]
    )

    return ModelArtifacts(
        pipeline=pipeline,
        feature_names=feature_names,
        train_report=train_report,
        test_report=test_report,
        confusion_matrix=cm,
        threshold=threshold,
    )


def evaluate_model(artifacts: ModelArtifacts) -> Dict[str, float]:
    """Extract top-level metrics for presentation."""

    test_report = artifacts.test_report
    return {
        "precision": test_report["1"]["precision"],
        "recall": test_report["1"]["recall"],
        "f1_score": test_report["1"]["f1-score"],
        "support": test_report["1"]["support"],
        "accuracy": test_report["accuracy"],
    }


def predict_failure_risk(
    pipeline: ClassifierMixin,
    data_row: Dict[str, float],
) -> Tuple[float, int]:
    """Predict probability of failure and recommended protective action."""

    df = pd.DataFrame([data_row])
    probability = pipeline.predict_proba(df)[0, 1]

    protective_action = int(
        (probability >= 0.5)
        or (data_row["wind_speed"] >= 23)
        or (data_row["vibration"] >= 9.0)
        or (data_row["icing_index"] >= 20)
    )
    return probability, protective_action
