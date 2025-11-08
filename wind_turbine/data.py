from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class TurbineSpec:
    """Physical characteristics of a wind turbine used to derive synthetic data."""

    rated_power_kw: float = 3500.0
    rotor_diameter_m: float = 110.0
    cut_in_speed: float = 3.5
    cut_out_speed: float = 25.0
    rated_speed: float = 12.0
    temp_operating_range: tuple[float, float] = (-20.0, 45.0)


def _logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_dataset(
    n_samples: int = 5000,
    *,
    seed: int | None = None,
    spec: TurbineSpec | None = None,
) -> pd.DataFrame:
    """Create a labelled dataset that mimics wind turbine telemetry.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    seed:
        Optional random seed for reproducibility.
    spec:
        Optional turbine specification. Defaults to `TurbineSpec()`.
    """

    rng = np.random.default_rng(seed)
    spec = spec or TurbineSpec()

    wind_speed = rng.gamma(shape=3.5, scale=3.0, size=n_samples)
    gust_factor = rng.normal(loc=1.05, scale=0.08, size=n_samples).clip(1.0, 1.4)
    rotor_speed = wind_speed * rng.normal(loc=4.5, scale=0.4, size=n_samples)
    vibration = rng.normal(loc=2.5, scale=0.8, size=n_samples) + (rotor_speed / 50) ** 1.2
    temp = rng.normal(loc=15, scale=8, size=n_samples)
    gearbox_oil_temp = temp + rng.normal(loc=18, scale=4, size=n_samples)
    humidity = rng.normal(loc=70, scale=12, size=n_samples).clip(20, 100)
    ambient_pressure = rng.normal(loc=1005, scale=12, size=n_samples)
    pitch_angle = rng.normal(loc=4, scale=6, size=n_samples) + np.maximum(
        0, wind_speed - spec.rated_speed
    )
    component_age = rng.uniform(low=0.3, high=15.0, size=n_samples)
    power_curve_efficiency = np.clip(
        (wind_speed / spec.rated_speed) ** 3, 0.0, 1.2
    )
    power_output = (
        0.5
        * 1.225
        * math.pi
        * (spec.rotor_diameter_m / 2) ** 2
        * wind_speed ** 3
        * 1e-3
        * power_curve_efficiency
    )
    power_output = np.clip(power_output, 0, spec.rated_power_kw * 1.1)

    icing_index = rng.beta(a=1.5, b=8, size=n_samples) * np.clip(
        100 - temp, 0, 60
    )
    electrical_load = power_output * rng.normal(loc=0.92, scale=0.04, size=n_samples)

    baseline_failure_logits = (
        0.03 * (wind_speed - spec.rated_speed)
        + 0.02 * (gearbox_oil_temp - temp)
        + 0.018 * (vibration - 2.5) ** 1.1
        + 0.01 * (pitch_angle - 7)
        + 0.05 * (electrical_load / spec.rated_power_kw)
        + 0.025 * (component_age / 15)
    )

    extreme_conditions = (wind_speed > spec.cut_out_speed * 0.92) | (
        vibration > 8.5
    )
    icing_risk = icing_index > 18

    failure_probability = _logistic(baseline_failure_logits)
    failure_probability += 0.25 * extreme_conditions + 0.18 * icing_risk
    failure_probability = np.clip(failure_probability, 0, 1)

    failures = rng.binomial(1, failure_probability)

    df = pd.DataFrame(
        {
            "wind_speed": wind_speed,
            "gust_factor": gust_factor,
            "rotor_speed": rotor_speed,
            "vibration": vibration,
            "ambient_temp": temp,
            "gearbox_oil_temp": gearbox_oil_temp,
            "humidity": humidity,
            "ambient_pressure": ambient_pressure,
            "pitch_angle": pitch_angle,
            "component_age_years": component_age,
            "power_output_kw": power_output,
            "electrical_load_kw": electrical_load,
            "icing_index": icing_index,
            "failure": failures,
            "failure_probability": failure_probability,
        }
    )
    df["protective_shutdown"] = (
        (failure_probability > 0.55)
        | (wind_speed > spec.cut_out_speed)
        | (vibration > 9.5)
    ).astype(int)

    return df


def derive_feature_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return feature matrix, target vector and feature names."""

    feature_names = [
        "wind_speed",
        "gust_factor",
        "rotor_speed",
        "vibration",
        "ambient_temp",
        "gearbox_oil_temp",
        "humidity",
        "ambient_pressure",
        "pitch_angle",
        "component_age_years",
        "power_output_kw",
        "electrical_load_kw",
        "icing_index",
    ]
    X = df[feature_names]
    y = df["failure"]
    return X, y, feature_names
