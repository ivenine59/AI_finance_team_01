import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import os

# Create directory for plots if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Define the assets
assets = ['^GSPC', 'TLT', 'ARKK']  # Example tickers for S&P 500, 20+ Year Treasuries, and ARK Innovation ETF

# Download data
data = yf.download(assets, start='2018-01-01', end='2023-12-31')['Adj Close']

risk_list = []
return_list = []
coefficients_list = []
dates = []

# Define the function to fit
def frontier_func(x, a, b, c):
    return a * np.sqrt(np.maximum(0, x - b)) + c

# Risk-free rate
risk_free_rate = 0.02

# Define the utility function
def utility_function(x, alpha):
    return alpha * x**2 + risk_free_rate

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

    # Generate a range of target returns within achievable range
    target_returns = np.linspace(min(expected_returns), max(expected_returns) - 0.0001, 100)
    frontier_returns, frontier_risks = [], []

    for r in target_returns:
        ef_temp = EfficientFrontier(expected_returns, cov_matrix)  # Create a new instance
        ef_temp.add_constraint(lambda w: w >= 0)
        ef_temp.efficient_return(r)
        ret, risk, _ = ef_temp.portfolio_performance()
        frontier_returns.append(ret)
        frontier_risks.append(risk)

    # Fit the curve to the efficient frontier points
    initial_guess = [1, min(frontier_risks), min(frontier_returns)]
    bounds = ([0, 0, 0], [10, max(frontier_risks), max(frontier_returns)])
    popt, _ = curve_fit(frontier_func, frontier_risks, frontier_returns, p0=initial_guess, bounds=bounds)
    popt = np.round(popt, 4)  # Round the coefficients to four decimal places
    coefficients_list.append(popt)
    dates.append(date)

    # Plot the efficient frontier
    fig, ax = plt.subplots(figsize=(12, 8))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False, show_fig=False)

    # Get the maximum Sharpe ratio portfolio without shorting
    ef = EfficientFrontier(expected_returns, cov_matrix)  # Recreate the object
    ef.add_constraint(lambda w: w >= 0)
    one_fund = ef.max_sharpe(risk_free_rate=0.02)
    ret, risk, sharpe = ef.portfolio_performance(risk_free_rate=0.02)
    risk_list.append(risk)
    return_list.append(ret)

    # Add the scatter plot of the maximum Sharpe ratio points
    ax.scatter(risk_list, return_list, marker='o', color='red', s=100)

    ax.get_legend().remove()
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Risk')
    ax.set_ylabel('Return')

    plt.tight_layout()
    plt.savefig(f'plots/efficient_frontier_{date.strftime("%Y-%m-%d")}.png')
    plt.close(fig)

# Convert coefficients list to a DataFrame for easier plotting
coefficients_df = pd.DataFrame(coefficients_list, columns=['a', 'b', 'c'], index=dates)

# Plot coefficients over time
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
coefficients_df['a'].plot(ax=axes[0], title='Coefficient a over time')
axes[0].set_ylabel('a')

coefficients_df['b'].plot(ax=axes[1], title='Coefficient b over time')
axes[1].set_ylabel('b')

coefficients_df['c'].plot(ax=axes[2], title='Coefficient c over time')
axes[2].set_ylabel('c')

plt.tight_layout()
plt.savefig('plots/coefficients_over_time.png')
plt.close(fig)

# Calculate and store the slope, intercept, and intersection points for each date
def tangent_line_slope_intercept(a, b, c, rf):
    # We need to find the point where the tangent line touches the efficient frontier
    # and passes through the risk-free rate. This requires solving for the tangent point.
    # We'll use the derivative of the efficient frontier function for this.
    
    # Solve the equation: a * sqrt(x - b) + c = slope * x + rf
    # First, find the tangent point where y = a * sqrt(x - b) + c is tangent to y = slope * x + rf
    
    def frontier_derivative(x):
        return a / (2 * np.sqrt(x - b))

    def objective(x):
        return (frontier_derivative(x) - (frontier_func(x, a, b, c) - rf) / x)**2
    
    result = minimize(objective, x0=0.1, bounds=[(b + 1e-6, None)])
    tangent_x = result.x[0]
    tangent_y = frontier_func(tangent_x, a, b, c)
    slope = (tangent_y - rf) / tangent_x
    intercept = rf
    return slope, intercept, tangent_x, tangent_y

# Solve for the intersection point of the utility function and the tangent line
def intersection_point(alpha, slope, intercept):
    # Solve alpha * x^2 + rf = slope * x + intercept
    # which is alpha * x^2 - slope * x + (rf - intercept) = 0
    coefficients = [alpha, -slope, (risk_free_rate - intercept)]
    roots = np.roots(coefficients)
    # We only want the positive root since risk (x) must be non-negative
    positive_roots = roots[roots >= 0]
    if len(positive_roots) == 0:
        return None
    x = positive_roots[0]
    y = slope * x + intercept
    return x, y

alpha = 1  # Example coefficient for the utility function

results = []

for date, coeffs in coefficients_df.iterrows():
    a, b, c = coeffs
    slope, intercept, tangent_x, tangent_y = tangent_line_slope_intercept(a, b, c, risk_free_rate)
    intersection = intersection_point(alpha, slope, intercept)
    if intersection is not None:
        results.append((date, slope, intercept, tangent_x, tangent_y, intersection[0], intersection[1]))

results_df = pd.DataFrame(results, columns=['Date', 'Slope', 'Intercept', 'Tangent_X', 'Tangent_Y', 'Intersection_X', 'Intersection_Y'])
results_df.set_index('Date', inplace=True)

# Plot slopes, intercepts, and intersection points over time
fig, axes = plt.subplots(4, 1, figsize=(12, 24))
results_df['Slope'].plot(ax=axes[0], title='Slope of Tangent Line over time')
axes[0].set_ylabel('Slope')

results_df['Intercept'].plot(ax=axes[1], title='Intercept of Tangent Line over time')
axes[1].set_ylabel('Intercept')

results_df['Tangent_X'].plot(ax=axes[2], title='Tangent X over time')
axes[2].set_ylabel('Tangent_X')

results_df['Intersection_X'].plot(ax=axes[3], title='Intersection X over time')
axes[3].set_ylabel('Intersection_X')

plt.tight_layout()
plt.savefig('plots/tangent_and_intersection_over_time.png')
plt.close(fig)

# Display the results
for date, row in results_df.iterrows():
    print(f"Date: {date.date()} - Slope: {row['Slope']:.4f}, Intercept: {row['Intercept']:.4f}, Tangent Point: ({row['Tangent_X']:.4f}, {row['Tangent_Y']:.4f}), Intersection: ({row['Intersection_X']:.4f}, {row['Intersection_Y']:.4f})")