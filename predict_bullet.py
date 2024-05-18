import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Define the assets
assets = ['^GSPC', 'TLT', 'ARKK']  # Example tickers for S&P 500, 20+ Year Treasuries, and ARK Innovation ETF

# Download data
data = yf.download(assets, start='2018-01-01', end='2023-12-31')['Adj Close']

# Calculate returns
returns = data.pct_change().dropna()

# Estimate expected returns and covariance matrix
expected_returns = returns.mean() * 252  # Annualized return
cov_matrix = returns.cov() * 252  # Annualized covariance

# Helper functions for efficient frontier
def portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_stddev

def generate_random_portfolios(num_portfolios, returns, cov_matrix):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_stddev = portfolio_performance(weights, returns, cov_matrix)
        
        results[0,i] = portfolio_stddev
        results[1,i] = portfolio_return
        results[2,i] = results[1,i] / results[0,i]
    
    return results, weights_record

# Generate portfolios
num_portfolios = 10000
results, weights = generate_random_portfolios(num_portfolios, expected_returns, cov_matrix)

# Plot the efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.show()

# ARIMA model prediction for each asset
predicted_returns = []

for asset in assets:
    model = ARIMA(returns[asset], order=(1, 1, 1))
    fit_model = model.fit()
    forecast = fit_model.forecast(steps=12)  # Predict next 12 months
    predicted_returns.append(forecast.mean())

predicted_returns = np.array(predicted_returns)

# Using historical covariance as a proxy for predicted period
predicted_cov_matrix = cov_matrix

# Generate portfolios for the predicted period
results, _ = generate_random_portfolios(num_portfolios, predicted_returns, predicted_cov_matrix)

# Plot the predicted efficient frontier along with historical ones
plt.figure(figsize=(10, 7))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', label='2024 Prediction')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier with 2024 Prediction')
plt.legend()
plt.show()
