"""Utility package for the wind turbine simulation Streamlit app."""

from .data import generate_synthetic_dataset
from .model import (
    build_training_pipeline,
    evaluate_model,
    predict_failure_risk,
    train_failure_model,
    ModelArtifacts,
)
from .performance import compute_energy_metrics, summarize_operating_modes
from .simulator import generate_operating_profile, simulate_sensor_stream

__all__ = [
    "generate_synthetic_dataset",
    "build_training_pipeline",
    "train_failure_model",
    "evaluate_model",
    "predict_failure_risk",
    "compute_energy_metrics",
    "summarize_operating_modes",
    "generate_operating_profile",
    "simulate_sensor_stream",
    "ModelArtifacts",
]
