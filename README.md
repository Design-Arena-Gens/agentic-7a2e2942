# Wind Turbine Reliability Lab

Interactive Streamlit application that simulates wind turbine telemetry, trains a machine learning failure protection model, and provides performance analytics dashboards.

## Features

- Synthetic SCADA-style dataset generator with configurable sample size and random seed
- Random forest based failure-risk classifier with confusion matrix, feature importance, and calibration controls
- Real-time protection simulation showing predicted risk over time and triggered shutdowns
- Interactive scenario testing to explore protective actions across operating conditions
- Performance analytics including energy KPIs and operating regime summaries
- Downloadable dataset for offline experimentation

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

Open the app in your browser at `http://localhost:8501`.

## Project Structure

```
├── app.py
├── requirements.txt
└── wind_turbine
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── performance.py
    └── simulator.py
```

## Deployment

The application is designed for Streamlit Cloud or similar Python hosting platforms. Ensure environment dependencies from `requirements.txt` are installed on the target platform.
