from __future__ import annotations

import numpy as np
import pandas as pd


def generate_operating_profile(
    *,
    steps: int = 96,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a diurnal operating profile capturing variability."""

    rng = np.random.default_rng(seed)
    time_index = pd.date_range("2024-01-01", periods=steps, freq="15min")
    base_wind = (
        8
        + 3 * np.sin(np.linspace(0, 2 * np.pi, steps))
        + rng.normal(0, 1.3, size=steps)
    )
    gust_factor = rng.normal(1.04, 0.05, size=steps).clip(1.0, 1.3)
    rotor_speed = base_wind * rng.normal(4.3, 0.25, size=steps)
    vibration = 2.5 + (rotor_speed / 60) ** 1.3 + rng.normal(0, 0.3, size=steps)
    ambient_temp = 12 + 4 * np.cos(np.linspace(0, 2 * np.pi, steps)) + rng.normal(
        0, 1.2, size=steps
    )
    icing_index = rng.beta(1.1, 12, size=steps) * np.clip(4 - ambient_temp, 0, 40)

    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "wind_speed": base_wind.clip(0, None),
            "gust_factor": gust_factor,
            "rotor_speed": rotor_speed.clip(0, None),
            "vibration": vibration.clip(0, None),
            "ambient_temp": ambient_temp,
            "icing_index": icing_index,
        }
    )
    return df


def simulate_sensor_stream(
    pipeline,
    profile: pd.DataFrame,
    *,
    rated_power_kw: float = 3500,
) -> pd.DataFrame:
    """Augment operating profile using the trained model for monitoring."""

    df = profile.copy()
    df["gearbox_oil_temp"] = df["ambient_temp"] + np.clip(
        20 + (df["rotor_speed"] / 70) ** 1.1, 15, 45
    )
    df["humidity"] = 65 + 15 * np.sin(np.linspace(0, 4 * np.pi, len(df)))
    df["ambient_pressure"] = 1008 + np.cos(np.linspace(0, 1.5 * np.pi, len(df))) * 6
    df["pitch_angle"] = np.clip(5 + (df["wind_speed"] - 12), -2, 25)
    df["component_age_years"] = 8.0
    df["power_output_kw"] = np.clip(
        rated_power_kw
        * ((df["wind_speed"] / 12).clip(lower=0, upper=1) ** 3),
        0,
        rated_power_kw,
    )
    df["electrical_load_kw"] = df["power_output_kw"] * 0.94

    probability = pipeline.predict_proba(
        df[
            [
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
        ]
    )[:, 1]
    df["predicted_failure_probability"] = probability
    df["protective_action"] = (
        (probability >= 0.5)
        | (df["wind_speed"] >= 23)
        | (df["vibration"] >= 9)
        | (df["icing_index"] >= 20)
    ).astype(int)
    return df
