from __future__ import annotations

import io
from textwrap import dedent

import pandas as pd
import plotly.express as px
import streamlit as st

from wind_turbine import (
    ModelArtifacts,
    compute_energy_metrics,
    generate_operating_profile,
    generate_synthetic_dataset,
    predict_failure_risk,
    simulate_sensor_stream,
    summarize_operating_modes,
    train_failure_model,
)

st.set_page_config(
    page_title="Wind Turbine Reliability Lab",
    layout="wide",
    page_icon="🌀",
)


def _format_metric(value: float, suffix: str = "", precision: int = 2) -> str:
    return f"{value:.{precision}f}{suffix}"


@st.cache_data(show_spinner=False)
def build_dataset(n_samples: int, seed: int):
    return generate_synthetic_dataset(n_samples, seed=seed)


def get_trained_model(df: pd.DataFrame, threshold: float, random_state: int) -> ModelArtifacts:
    cache_key = f"{len(df)}-{threshold:.2f}-{random_state}"
    cache = st.session_state.setdefault("_model_cache", {})
    artifacts = cache.get(cache_key)
    if artifacts is None:
        with st.spinner("Training protection model..."):
            artifacts = train_failure_model(
                df, threshold=threshold, random_state=random_state
            )
        cache[cache_key] = artifacts
    return artifacts


def render_header():
    st.title("Wind Turbine Failure Protection & Performance Lab")
    st.caption(
        "Synthetic digital twin to explore turbine telemetry, proactive protection logic, "
        "and machine learning driven risk scoring."
    )


def render_sidebar() -> tuple[int, int, float]:
    with st.sidebar:
        st.header("Simulation Controls")
        n_samples = st.slider("Synthetic sample size", 1500, 10000, 5500, step=500)
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=1234)
        threshold = st.slider(
            "Protection trigger threshold",
            min_value=0.2,
            max_value=0.9,
            value=0.55,
            step=0.05,
        )
        st.markdown(
            dedent(
                """
                *Lower thresholds react faster but might create more false alarms. Higher thresholds
                delay shutdowns but risk missing emerging failures.*
                """
            )
        )
        return n_samples, seed, threshold


def render_model_metrics(artifacts: ModelArtifacts, df: pd.DataFrame):
    metrics = artifacts.test_report
    st.subheader("Model Validation Snapshot")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", _format_metric(metrics["accuracy"]))
    metric_cols[1].metric("Recall (Failure)", _format_metric(metrics["1"]["recall"]))
    metric_cols[2].metric("Precision (Failure)", _format_metric(metrics["1"]["precision"]))
    metric_cols[3].metric("F1 (Failure)", _format_metric(metrics["1"]["f1-score"]))

    importance = artifacts.pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame(
        {"feature": artifacts.feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    fig_importance = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature importance ranking",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_importance.update_layout(height=450, yaxis_categoryorder="total ascending")

    confusion = artifacts.confusion_matrix
    cm_fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="GnBu",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion matrix",
    )
    cm_fig.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["Normal", "Failure"])
    cm_fig.update_yaxes(tickmode="array", tickvals=[0, 1], ticktext=["Normal", "Failure"])

    col_left, col_right = st.columns((1, 1))
    col_left.plotly_chart(fig_importance, use_container_width=True)
    col_right.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("#### Dataset Distribution")
    hist_cols = st.columns(3)
    with hist_cols[0]:
        fig = px.histogram(df, x="wind_speed", nbins=35, title="Wind speed distribution")
        st.plotly_chart(fig, use_container_width=True)
    with hist_cols[1]:
        fig = px.histogram(df, x="vibration", nbins=35, title="Vibration index")
        st.plotly_chart(fig, use_container_width=True)
    with hist_cols[2]:
        fig = px.histogram(
            df, x="failure_probability", color="failure", nbins=35, title="Failure propensity"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_live_simulation(artifacts: ModelArtifacts, seed: int):
    st.subheader("Real-time Protection Simulation")
    profile = generate_operating_profile(seed=seed)
    simulated = simulate_sensor_stream(artifacts.pipeline, profile)

    prob_fig = px.line(
        simulated,
        x="timestamp",
        y="predicted_failure_probability",
        color="protective_action",
        color_discrete_map={0: "#1f77b4", 1: "#d62728"},
        labels={"protective_action": "Protection Triggered"},
        title="Predicted failure probability over time",
    )
    prob_fig.add_hline(y=artifacts.threshold, line_dash="dash", line_color="gray")
    st.plotly_chart(prob_fig, use_container_width=True)

    st.markdown("##### Protective Actions")
    alert_events = simulated[simulated["protective_action"] == 1]
    if alert_events.empty:
        st.success("No protective shutdowns triggered in the simulated period.")
    else:
        st.dataframe(
            alert_events[
                [
                    "timestamp",
                    "wind_speed",
                    "vibration",
                    "icing_index",
                    "predicted_failure_probability",
                ]
            ],
            hide_index=True,
        )


def render_performance_insights(df: pd.DataFrame):
    st.subheader("Performance & Energy KPIs")
    metrics = compute_energy_metrics(df)
    kpi_cols = st.columns(4)
    kpi_cols[0].metric(
        "Total energy (MWh)", _format_metric(metrics["energy_mwh"], precision=1)
    )
    kpi_cols[1].metric(
        "Avg capacity factor",
        _format_metric(metrics["avg_capacity_factor"] * 100, suffix="%", precision=1),
    )
    kpi_cols[2].metric("Mean wind speed (m/s)", _format_metric(metrics["avg_wind_speed"]))
    kpi_cols[3].metric(
        "Protection engagement",
        _format_metric(metrics["protective_shutdown_rate"] * 100, suffix="%", precision=1),
    )

    regime_summary = summarize_operating_modes(df)
    regime_fig = px.bar(
        regime_summary,
        x="operating_regime",
        y="avg_failure_risk",
        color="avg_power_kw",
        title="Operating mode risk profile",
        labels={"avg_failure_risk": "Mean failure risk", "avg_power_kw": "Avg power (kW)"},
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(regime_fig, use_container_width=True)
    st.dataframe(regime_summary, hide_index=True)


def render_interactive_scenario(artifacts: ModelArtifacts, df: pd.DataFrame):
    st.subheader("Interactive Scenario Testing")
    col1, col2, col3 = st.columns(3)
    with col1:
        wind_speed = st.slider("Wind speed (m/s)", 0.0, 30.0, 12.0, step=0.5)
        rotor_speed = st.slider("Rotor speed (rpm)", 0.0, 200.0, float(12 * 4.5), step=1.0)
        vibration = st.slider("Vibration index", 0.0, 12.0, 5.0, step=0.1)
        icing_index = st.slider("Blade icing index", 0.0, 40.0, 5.0, step=0.5)
    with col2:
        ambient_temp = st.slider("Ambient temperature (°C)", -30.0, 50.0, 10.0, step=0.5)
        gearbox = st.slider(
            "Gearbox oil temperature (°C)", -10.0, 120.0, ambient_temp + 25, step=0.5
        )
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0, step=1.0)
        pressure = st.slider("Ambient pressure (hPa)", 950.0, 1040.0, 1008.0, step=0.5)
    with col3:
        pitch_angle = st.slider("Pitch angle (°)", -5.0, 30.0, 6.0, step=0.5)
        component_age = st.slider("Component age (years)", 0.1, 25.0, 7.5, step=0.1)
        power_output = st.slider("Instantaneous power (kW)", 0.0, 4000.0, 1800.0, step=10.0)
        electrical_load = st.slider(
            "Electrical load (kW)", 0.0, 4000.0, power_output * 0.95, step=10.0
        )

    scenario = {
        "wind_speed": wind_speed,
        "gust_factor": 1.05,
        "rotor_speed": rotor_speed,
        "vibration": vibration,
        "ambient_temp": ambient_temp,
        "gearbox_oil_temp": gearbox,
        "humidity": humidity,
        "ambient_pressure": pressure,
        "pitch_angle": pitch_angle,
        "component_age_years": component_age,
        "power_output_kw": power_output,
        "electrical_load_kw": electrical_load,
        "icing_index": icing_index,
    }
    probability, protective_action = predict_failure_risk(artifacts.pipeline, scenario)
    st.metric("Failure probability", _format_metric(probability * 100, suffix="%", precision=1))

    if protective_action:
        st.error(
            "⚠️ Protective shutdown recommended. "
            "Risk exceeds calibrated threshold or critical limits."
        )
    else:
        st.success("✅ Operating within safe band. No immediate protective action required.")

    nearest_points = (
        df.assign(
            risk_gap=(df["failure_probability"] - probability) ** 2,
        )
        .nsmallest(200, "risk_gap")
    )
    scatter = px.scatter(
        nearest_points,
        x="wind_speed",
        y="power_output_kw",
        color="failure",
        size="vibration",
        hover_data={
            "wind_speed": True,
            "power_output_kw": True,
            "vibration": True,
            "failure_probability": ":.2f",
            "failure": True,
        },
        title="Neighbourhood comparison of similar conditions",
        labels={"failure": "Observed failure"},
    )
    st.plotly_chart(scatter, use_container_width=True)


def render_dataset_download(df: pd.DataFrame):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download synthetic dataset (CSV)",
        file_name="wind_turbine_data.csv",
        mime="text/csv",
        data=csv_buffer.getvalue(),
    )


def main():
    render_header()
    n_samples, seed, threshold = render_sidebar()
    df = build_dataset(n_samples, seed)
    artifacts = get_trained_model(df, threshold, random_state=seed)

    overview_tab, protection_tab, performance_tab, data_tab = st.tabs(
        ["Model Overview", "Protection System", "Performance Analytics", "Raw Data"]
    )

    with overview_tab:
        render_model_metrics(artifacts, df)

    with protection_tab:
        render_live_simulation(artifacts, seed)
        render_interactive_scenario(artifacts, df)

    with performance_tab:
        render_performance_insights(df)

    with data_tab:
        st.write("Preview of generated telemetry dataset")
        st.dataframe(df.head(200), height=400)
        render_dataset_download(df)


if __name__ == "__main__":
    main()
