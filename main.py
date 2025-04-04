from src.load_data import load_stock_data

data_dir = "./data"
print("Loading stock data...")  # Confirm script is running

stocks = load_stock_data(data_dir)

# Print the first few rows of each stock
for name, df in stocks.items():
    print(f"\n{name} Data:\n", df.head())

from src.load_data import load_stock_data
from src.analysis import calculate_log_returns

data_dir = "./data"
print("Loading stock data...")
stocks = load_stock_data(data_dir)

print("Calculating log returns...")
log_returns = calculate_log_returns(stocks)

print(log_returns.head())

from src.simulation import simulate_portfolios

# Step 3: Run Monte Carlo Simulation
print("Running Monte Carlo simulation...")
simulated = simulate_portfolios(log_returns, num_portfolios=5000)

# Step 4: Show top results
print(simulated.sort_values(by="Sharpe", ascending=False).head())

import matplotlib.pyplot as plt
from src.simulation import get_optimal_portfolio

# Step 5: Find and show optimal portfolio
optimal = get_optimal_portfolio(simulated)
print("\nOptimal Portfolio (Max Sharpe):")
print(optimal)

# Step 6: Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(simulated['Volatility'], simulated['Returns'], c=simulated['Sharpe'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier - Monte Carlo Simulation')

# Highlight the optimal portfolio
plt.scatter(optimal['Volatility'], optimal['Returns'], color='red', marker='*', s=200, label='Max Sharpe')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from src.analysis import calculate_var

# Step 7: Calculate Value at Risk (VaR)
portfolio_weights = optimal['Weights']
var_95 = calculate_var(log_returns, portfolio_weights, confidence_level=0.95)
var_99 = calculate_var(log_returns, portfolio_weights, confidence_level=0.99)

print(f"\nEstimated Daily Value at Risk (VaR):")
print(f"95% Confidence: £{var_95:,.2f}")
print(f"99% Confidence: £{var_99:,.2f}")
