# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import resample
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile, os, io

st.set_page_config(page_title="Renewable Energy Forecasting", layout="wide")

# ====================================================
# 1. Title
# ====================================================
st.title("Renewable Energy Forecasting with Uncertainty (Demo)")
st.write("Predict solar output using past data with uncertainty estimation.")

# ====================================================
# 2. Data Input
# ====================================================
st.sidebar.header("Data Options")
data_option = st.sidebar.radio("Choose dataset:", ["Simulated Data", "Upload CSV"])

if data_option == "Simulated Data":
    n_points = st.sidebar.slider("Number of points", 200, 1000, 500)
    np.random.seed(42)
    time = np.arange(n_points)
    signal = 10 + 5*np.sin(time * 2*np.pi/50)
    noise = np.random.normal(0, 1, n_points)
    base_solar = signal + noise
    df_base = pd.DataFrame({"time": time, "solar_output": base_solar})
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df_base = pd.read_csv(uploaded_file)
        num_cols = df_base.select_dtypes(include=np.number).columns
        if "solar_output" not in df_base.columns:
            df_base = df_base.rename(columns={num_cols[0]: "solar_output"})
    else:
        st.stop()

st.subheader("Data Preview")
st.dataframe(df_base.head())

# ====================================================
# 3. Forecast Parameters
# ====================================================
st.sidebar.header("Forecast Parameters")
window_size = st.sidebar.slider("Window Size (past points)", 5, 50, 24)
horizon = st.sidebar.slider("Forecast Horizon (future points)", 1, 24, 6)
n_bootstrap = st.sidebar.slider("Bootstrap Samples", 10, 100, 50)

# ====================================================
# 4. Model Selection
# ====================================================
st.sidebar.header("Model Options")
model_choice = st.sidebar.selectbox("Choose Model:", ["Linear Regression", "LSTM", "GRU", "XGBoost"])

if model_choice in ["LSTM", "GRU"]:
    epochs = st.sidebar.slider("Epochs", 1, 50, 10)
    batch_size = st.sidebar.slider("Batch Size", 8, 64, 16)
    neurons = st.sidebar.slider("Neurons per layer", 4, 128, 32)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)

# ====================================================
# 5. Sliding Window Function
# ====================================================
def create_windowed_data(series, window_size=24, horizon=1):
    X, y = [], []
    n = len(series)
    for i in range(n - window_size - horizon + 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size:i+window_size+horizon])
    X = np.array(X)
    y = np.array(y)
    if horizon == 1:
        y = y.reshape(-1, 1)
    return X, y

# ====================================================
# 6. Multi-Scenario Forecast Function
# ====================================================
def forecast_scenario(series, scenario_name="Normal"):
    multiplier = {"Normal":1.0, "Sunny":1.2, "Cloudy":0.7}[scenario_name]
    series_adj = series * multiplier
    X, y = create_windowed_data(series_adj, window_size, horizon)
    split_idx = int(0.8*len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if model_choice in ["LSTM", "GRU"]:
        X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    else:
        X_train_rnn, X_test_rnn = X_train, X_test

    # Model Training
    if model_choice == "Linear Regression":
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_train_rnn, y_train)
        y_pred = model.predict(X_test_rnn)

    elif model_choice in ["LSTM", "GRU"]:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        if model_choice == "LSTM":
            model.add(LSTM(neurons, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        else:
            model.add(GRU(neurons, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(horizon))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        model.fit(X_train_rnn, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X_test_rnn)

    elif model_choice == "XGBoost":
        import xgboost as xgb
        y_pred = np.zeros_like(y_test)
        for i in range(horizon):
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            xgb_model.fit(X_train_rnn, y_train[:,i])
            y_pred[:,i] = xgb_model.predict(X_test_rnn)

    # Bootstrap Uncertainty
    if model_choice in ["Linear Regression","XGBoost"]:
        preds = []
        for _ in range(n_bootstrap):
            X_bs, y_bs = resample(X_train_rnn, y_train, replace=True)
            if model_choice == "Linear Regression":
                m_bs = MultiOutputRegressor(LinearRegression())
                m_bs.fit(X_bs, y_bs)
                preds.append(m_bs.predict(X_test_rnn))
            else:
                y_bs_pred = np.zeros_like(y_test)
                for j in range(horizon):
                    xgb_bs = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
                    xgb_bs.fit(X_bs, y_bs[:,j])
                    y_bs_pred[:,j] = xgb_bs.predict(X_test_rnn)
                preds.append(y_bs_pred)
        preds = np.array(preds)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
    else:
        mean_pred = y_pred
        std_pred = np.zeros_like(y_pred)

    rmse = np.sqrt(np.mean((y_test - mean_pred)**2, axis=0))
    mae = np.mean(np.abs(y_test - mean_pred), axis=0)

    return y_test, mean_pred, std_pred, rmse, mae

# ====================================================
# 7. Multi-Scenario Forecast Computation (Before Visualization)
# ====================================================
st.sidebar.header("Select Scenarios to Compare")
selected_scenarios = st.sidebar.multiselect(
    "Choose one or more scenarios", ["Normal","Sunny","Cloudy"], default=["Normal"]
)

# Precompute forecasts for all selected scenarios
scenario_results = {}  # store all results
for scenario_name in selected_scenarios:
    y_test_s, mean_pred_s, std_pred_s, rmse_s, mae_s = forecast_scenario(
        df_base["solar_output"].values, scenario_name
    )
    scenario_results[scenario_name] = {
        "y_test": y_test_s,
        "mean_pred": mean_pred_s,
        "std_pred": std_pred_s,
        "rmse": rmse_s,
        "mae": mae_s
    }

# ====================================================
# 8. Evaluation / Metrics
# ====================================================
st.subheader("Forecast Metrics")
for scenario_name in selected_scenarios:
    res = scenario_results[scenario_name]
    st.write(f"### Scenario: {scenario_name}")
    st.write("RMSE per step:", res["rmse"])
    st.write("MAE per step:", res["mae"])


# ============================
# 9. Visualization
# ============================
st.subheader("Forecast Visualization (Multi-Scenario)")
overlay_steps = st.checkbox("Overlay all horizon steps", value=True)
fig = go.Figure()
true_colors = px.colors.qualitative.Plotly
pred_colors = px.colors.qualitative.Dark24

for scenario_idx, scenario_name in enumerate(selected_scenarios):
    res = scenario_results[scenario_name]
    if overlay_steps:
        for step in range(horizon):
            true_color = true_colors[(scenario_idx*horizon + step) % len(true_colors)]
            pred_color = pred_colors[(scenario_idx*horizon + step) % len(pred_colors)]
            fig.add_trace(go.Scatter(
                y=res["y_test"][:, step],
                mode='lines',
                name=f"{scenario_name} True Step {step+1}",
                line=dict(color=true_color, width=2)
            ))
            fig.add_trace(go.Scatter(
                y=res["mean_pred"][:, step],
                mode='lines',
                name=f"{scenario_name} Pred Step {step+1}",
                line=dict(color=pred_color, width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                y=res["mean_pred"][:, step]+2*res["std_pred"][:,step],
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                y=res["mean_pred"][:, step]-2*res["std_pred"][:,step],
                mode='lines', fill='tonexty',
                fillcolor='rgba(255,165,0,0.15)',
                line=dict(width=0),
                name=f"{scenario_name} Uncertainty ±2σ" if step==0 else None,
                showlegend=True if step==0 else False
            ))
    else:
        step_to_plot = st.slider(
            f"Select horizon step to plot ({scenario_name})",
            1, horizon, 1,
            key=f"step_slider_{scenario_name}"
        ) - 1
        true_color = true_colors[scenario_idx % len(true_colors)]
        pred_color = pred_colors[scenario_idx % len(pred_colors)]
        fig.add_trace(go.Scatter(
            y=res["y_test"][:, step_to_plot],
            mode='lines',
            name=f"{scenario_name} True Step {step_to_plot+1}",
            line=dict(color=true_color, width=2)
        ))
        fig.add_trace(go.Scatter(
            y=res["mean_pred"][:, step_to_plot],
            mode='lines',
            name=f"{scenario_name} Pred Step {step_to_plot+1}",
            line=dict(color=pred_color, width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            y=res["mean_pred"][:, step_to_plot]+2*res["std_pred"][:, step_to_plot],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            y=res["mean_pred"][:, step_to_plot]-2*res["std_pred"][:, step_to_plot],
            mode='lines', fill='tonexty',
            fillcolor='rgba(255,165,0,0.15)',
            line=dict(width=0),
            name=f"{scenario_name} Uncertainty ±2σ",
            showlegend=True
        ))

fig.update_layout(
    title="Forecast vs True Solar Output (Multi-Scenario Overlay)",
    xaxis_title="Time (test window index)",
    yaxis_title="Solar Output",
    hovermode="x unified"
)

# Display the figure
st.plotly_chart(fig, use_container_width=True)



# ====================================================
# 10. Error Heatmap (Crucial Graph)
# ====================================================
import seaborn as sns

st.subheader("Error Distribution Heatmap")

for scenario_name in selected_scenarios:
    res = scenario_results[scenario_name]
    errors = res["y_test"] - res["mean_pred"]
    error_df = pd.DataFrame(errors, columns=[f"Step {i+1}" for i in range(errors.shape[1])])
    
    fig_err, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(error_df.T, cmap="RdBu_r", center=0, cbar_kws={'label': 'Error'})
    ax.set_title(f"Forecast Error Heatmap – {scenario_name}")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Forecast Horizon Step")
    st.pyplot(fig_err)

# ====================================================
# 11. Animated GIF of Forecast Evolution
# ====================================================
import matplotlib.animation as animation

scenario_name = selected_scenarios[0]  # take first scenario
res = scenario_results[scenario_name]

fig_anim, ax_anim = plt.subplots()
line_true, = ax_anim.plot([], [], lw=2, label="True Solar Output", color="blue")
line_pred, = ax_anim.plot([], [], lw=2, ls='--', label="Predicted Solar Output", color="orange")

ax_anim.set_xlim(0, len(res["y_test"]))
ax_anim.set_ylim(
    min(res["y_test"].min(), res["mean_pred"].min())-2,
    max(res["y_test"].max(), res["mean_pred"].max())+2
)
ax_anim.set_title(f"Forecast Evolution – {scenario_name} Scenario", fontsize=14)
ax_anim.set_xlabel("Test Sample Index", fontsize=12)
ax_anim.set_ylabel("Solar Output", fontsize=12)
ax_anim.legend()

def init():
    line_true.set_data([], [])
    line_pred.set_data([], [])
    return line_true, line_pred

def animate(i):
    x = np.arange(i+1)
    line_true.set_data(x, res["y_test"][:i+1, 0])   # first horizon step
    line_pred.set_data(x, res["mean_pred"][:i+1, 0])
    return line_true, line_pred

ani = animation.FuncAnimation(fig_anim, animate, init_func=init,
                              frames=len(res["y_test"]), interval=200, blit=True)

# Save GIF
gif_path = os.path.join(tempfile.gettempdir(), "forecast_animation.gif")
ani.save(gif_path, writer="pillow")

st.image(gif_path, caption=f"Forecast Evolution Animation – {scenario_name}")
