import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import imageio

# 자산 정의
assets = ['^GSPC', 'TLT', 'ARKK']  # 예시 티커: S&P 500, 20+년 국채, ARK 혁신 ETF

# 데이터 다운로드
data = yf.download(assets, start='2018-01-01', end='2023-12-31')['Adj Close']

# 함수 정의
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_stddev
    return portfolio_return, portfolio_stddev, sharpe_ratio

def generate_random_portfolios(num_portfolios, returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_stddev, sharpe_ratio = portfolio_performance(weights, returns, cov_matrix, risk_free_rate)
        
        results[0,i] = portfolio_stddev
        results[1,i] = portfolio_return
        results[2,i] = sharpe_ratio
    
    return results, weights_record

# GIF 생성 준비
images = []

# x축, y축 범위 설정
x_min, x_max = 0, 0.4
y_min, y_max = -0.8, 1

# 월별 Efficient Frontier 생성 및 시각화
for date in pd.date_range(start='2018-01-01', end='2023-12-31', freq='M'):
    monthly_data = data[:date]
    returns = monthly_data.pct_change().dropna()
    expected_returns = returns.mean() * 252  # 연간 수익률
    cov_matrix = returns.cov() * 252  # 연간 공분산
    risk_free_rate = 0.02  # 예시 무위험 수익률, 2%
    
    num_portfolios = 10000
    results, weights = generate_random_portfolios(num_portfolios, expected_returns, cov_matrix, risk_free_rate)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title(f'Efficient Frontier as of {date.strftime("%Y-%m")}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(f'efficient_frontier_{date.strftime("%Y-%m")}.png')
    images.append(imageio.imread(f'efficient_frontier_{date.strftime("%Y-%m")}.png'))
    plt.close()

# GIF 저장
imageio.mimsave('efficient_frontier.gif', images, duration=0.5)
