import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting

# Define the assets
assets = ['^GSPC', 'TLT', 'ARKK']  # Example tickers for S&P 500, 20+ Year Treasuries, and ARK Innovation ETF

# Download data
data = yf.download(assets, start='2018-01-01', end='2023-12-31')['Adj Close']

risk_list = []
return_list = []

fig, ax = plt.subplots(figsize=(12, 8))

for date in pd.date_range(start='2019-01-01', end='2023-12-31', freq='M'):
    monthly_data = data[:date]

    # Calculate returns and risks
    returns = monthly_data.pct_change().dropna()
    expected_returns = np.array(returns.mean() * 252)  # Annualized return
    cov_matrix = np.array(returns.cov() * 252)  # Annualized covariance

    # Create Efficient Frontier instance
    ef = EfficientFrontier(expected_returns, cov_matrix)

    # Disallow shorting by setting weight bounds to (0, 1) for all assets
    ef.add_constraint(lambda w: w >= 0)

    # Plot the efficient frontier
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False, show_fig=False)

    # Get the maximum Sharpe ratio portfolio without shorting
    ef = EfficientFrontier(expected_returns, cov_matrix)  # Recreate the object
    ef.add_constraint(lambda w: w >= 0)
    one_fund = ef.max_sharpe()
    ret, risk, sharpe = ef.portfolio_performance()
    risk_list.append(risk)
    return_list.append(ret)

# Add the scatter plot of the maximum Sharpe ratio points
ax.scatter(risk_list, return_list, marker='o', color='red', s=100)

ax.get_legend().remove()
ax.set_title('Efficient Frontier')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')

plt.tight_layout()
plt.show()