import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.load_data import load_stock_data
from src.analysis import calculate_log_returns, calculate_var
from src.simulation import simulate_portfolios, get_optimal_portfolio

# --- App Title ---
st.title("📈 AI-Powered Stock Portfolio Optimiser")

# --- Parameters ---
st.sidebar.header("Simulation Settings")
num_portfolios = st.sidebar.slider("Number of portfolios to simulate", 1000, 10000, 5000, step=500)
confidence = st.sidebar.selectbox("VaR Confidence Level", [0.95, 0.99])
holding_period = st.sidebar.slider("VaR Holding Period (days)", 1, 10, 1)

# --- Load Data ---
data_dir = "./data"
st.write("Loading and processing stock data...")
stocks = load_stock_data(data_dir)
log_returns = calculate_log_returns(stocks)

# --- Run Simulation ---
st.write("Running Monte Carlo simulation...")
simulation = simulate_portfolios(log_returns, num_portfolios=num_portfolios)
optimal = get_optimal_portfolio(simulation)

# --- Display Results ---
st.subheader("📊 Optimal Portfolio")
st.write(f"**Expected Return:** {optimal['Returns']:.2%}")
st.write(f"**Volatility:** {optimal['Volatility']:.2%}")
st.write(f"**Sharpe Ratio:** {optimal['Sharpe']:.2f}")

# --- Value at Risk ---
var = calculate_var(log_returns, optimal['Weights'], confidence_level=confidence, holding_period=holding_period)
st.subheader("💥 Value at Risk (VaR)")
st.write(f"{int(confidence*100)}% confidence over {holding_period} day(s): **£{var:.2f} per £1 invested**")

# --- Portfolio Weights ---
weights = pd.Series(optimal['Weights'], index=log_returns.columns)
st.subheader("📦 Portfolio Allocation")
st.bar_chart(weights)

# --- Efficient Frontier ---
st.subheader("⚖️ Efficient Frontier")

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(simulation['Volatility'], simulation['Returns'], c=simulation['Sharpe'], cmap='viridis', alpha=0.6)
ax.scatter(optimal['Volatility'], optimal['Returns'], color='red', marker='*', s=200, label='Max Sharpe')
plt.colorbar(sc, label='Sharpe Ratio')
ax.set_xlabel('Volatility (Risk)')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier')
ax.legend()
st.pyplot(fig)
