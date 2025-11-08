from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_energy_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate production and utilisation metrics."""

    total_energy_mwh = df["power_output_kw"].sum() / 1000
    avg_capacity_factor = (
        df["power_output_kw"] / df["power_output_kw"].max().clip(lower=1)
    ).mean()
    avg_wind_speed = df["wind_speed"].mean()
    shutdown_rate = df["protective_shutdown"].mean()

    return {
        "energy_mwh": total_energy_mwh,
        "avg_capacity_factor": avg_capacity_factor,
        "avg_wind_speed": avg_wind_speed,
        "protective_shutdown_rate": shutdown_rate,
    }


def summarize_operating_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Classify regime of operation to enrich dashboard insights."""

    conditions = [
        df["wind_speed"] < 4,
        (df["wind_speed"] >= 4) & (df["wind_speed"] <= 15) & (df["vibration"] < 7.5),
        (df["wind_speed"] > 15) & (df["wind_speed"] <= 23),
        (df["wind_speed"] > 23) | (df["vibration"] >= 8.5),
    ]
    choices = ["idle", "optimal", "high_load", "extreme"]
    regimes = np.select(conditions, choices, default="unknown")

    grouped = (
        df.assign(operating_regime=regimes)
        .groupby("operating_regime")
        .agg(
            sample_count=("operating_regime", "count"),
            avg_failure_risk=("failure_probability", "mean"),
            avg_power_kw=("power_output_kw", "mean"),
            shutdown_rate=("protective_shutdown", "mean"),
        )
        .reset_index()
        .sort_values("avg_failure_risk", ascending=False)
    )
    return grouped
