import pandas as pd
import numpy as np

def calculate_log_returns(price_data):
    """
    Extracts 'Close' prices, ensures unique timestamps, 
    aligns them on common timestamps, and calculates log returns.
    """
    close_prices = pd.DataFrame()

    for name, df in price_data.items():
        df = df[['Close']].copy()

        # Ensure index (Datetime) is unique
        df = df[~df.index.duplicated(keep='first')]

        df.rename(columns={'Close': name}, inplace=True)
        close_prices = pd.concat([close_prices, df], axis=1)

    # Drop rows with any missing data
    close_prices.dropna(inplace=True)

    # Calculate log returns
    log_returns = np.log(close_prices / close_prices.shift(1))
    return log_returns.dropna()

import scipy.stats as stats

def calculate_var(log_returns, weights, confidence_level=0.95, holding_period=1):
    """
    Calculates the parametric VaR (Value at Risk) for a portfolio.
    """
    # Convert weights list to numpy array
    weights = np.array(weights)

    # Annualised mean and covariance
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

    # Portfolio mean and std dev
    portfolio_mean = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # z-score for given confidence level
    z_score = stats.norm.ppf(1 - confidence_level)

    # Daily VaR (scaled to holding period)
    var = -(portfolio_mean * holding_period + z_score * portfolio_std * np.sqrt(holding_period))

    return var
