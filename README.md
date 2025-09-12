# Renewable Energy Forecasting with Uncertainty

This repository contains a **Streamlit-based demo** for forecasting solar energy output using historical data. The application supports multi-scenario forecasts, uncertainty estimation via bootstrap, and interactive visualization with Plotly.

> **Note:** This demo was developed with the assistance of GPT-5 Mini for code structure, planning, and documentation.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Forecasting Models](#forecasting-models)
* [Visualization](#visualization)
* [Future Improvements](#future-improvements)
* [License](#license)

---

## Features

* Load data via **simulated time series** or **CSV upload**.
* Configurable **forecast parameters**:

  * Window size (past points)
  * Forecast horizon (future points)
  * Number of bootstrap samples
* Supports multiple forecasting **models**:

  * Linear Regression (multi-output)
  * LSTM
  * GRU
  * XGBoost
* Multi-scenario forecasting:

  * Normal
  * Sunny
  * Cloudy
* Bootstrap-based **uncertainty estimation** (mean ± standard deviation)
* Computes **forecast metrics**: RMSE and MAE per horizon step
* Interactive **Plotly visualization**:

  * Multi-scenario overlay
  * Horizon step selection
  * Uncertainty bands

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/renewable-energy-forecasting.git
cd renewable-energy-forecasting
```

2. Create a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* `streamlit`
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `plotly`
* `tensorflow` (for LSTM/GRU)
* `xgboost`
* `fpdf`

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and interact with the app:

* Choose dataset (simulated or CSV)
* Set forecast parameters in the sidebar
* Select model and scenarios
* View metrics and interactive Plotly charts

---

## Project Structure

```
renewable-energy-forecasting/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── example_plotly.png    # Placeholder for exported Plotly figure
└── data/                 # Optional folder for CSV datasets
```

---

## Forecasting Models

* **Linear Regression**: Multi-output regression for each horizon step.
* **LSTM / GRU**: Recurrent neural networks for sequential prediction.
* **XGBoost**: Gradient boosting regression per horizon step.
* **Bootstrap**: Resampling to quantify uncertainty for non-neural models.

---

## Visualization

* Multi-scenario overlay using Plotly
* True vs Predicted values
* Uncertainty bands (±2σ)
* Slider for selecting individual horizon steps
* Planned PDF report integration using exported Plotly images

---

## Future Improvements

* Embed Plotly figures directly into PDF reports
* Support additional renewable energy datasets (wind, hydro)
* Include hyperparameter tuning for all models
* Enable batch export of scenario forecasts

---

## License

This project is released under the MIT License.
