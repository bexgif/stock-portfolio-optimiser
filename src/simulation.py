import numpy as np
import pandas as pd

def simulate_portfolios(log_returns, num_portfolios=10000, risk_free_rate=0.01):
    np.random.seed(42)  # for reproducibility

    # Annualise returns and covariance
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    num_assets = len(mean_returns)

    results = {
        "Returns": [],
        "Volatility": [],
        "Sharpe": [],
        "Weights": []
    }

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results["Returns"].append(portfolio_return)
        results["Volatility"].append(portfolio_volatility)
        results["Sharpe"].append(sharpe_ratio)
        results["Weights"].append(weights)

    return pd.DataFrame(results)

def get_optimal_portfolio(simulated_df):
    """
    Returns the portfolio with the highest Sharpe ratio.
    """
    return simulated_df.loc[simulated_df['Sharpe'].idxmax()]
