import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import imageio.v2 as imageio  # 이미지 읽기를 위해 imageio.v2 사용
import os
from sklearn.metrics.pairwise import cosine_similarity
from kofr import ret_kofr

# 자산 정의
assets = ['^GSPC', 'TLT', '^KS11']  # 예시 티커: S&P 500, 20+년 국채, KOSPI

# 데이터 다운로드
data = yf.download(assets, start='2018-01-01', end='2023-12-31')['Adj Close']

# 결과 저장할 디렉토리 생성
output_dir = 'efficient_frontier_images'
os.makedirs(output_dir, exist_ok=True)

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

def get_closest_date(date, df):
    """
    Args:
        date: The target date.
        df: The DataFrame to search.

    Returns:
        The closest date in the DataFrame that is after the target date.
    """
    return df.index[df.index >= date].min()

# GIF 생성 준비
images = []

# x축, y축 범위 설정
x_min, x_max = 0, 0.4
y_min, y_max = -0.8, 1

# Sharpe Ratio가 가장 높은 포트폴리오의 weight 저장
best_weights_record = {}

# 이전 달의 가중치를 저장할 변수 초기화
previous_weights = None

# Risk-free DataFrame
df_kofr = ret_kofr(bool = True)

# 월별 Efficient Frontier 생성 및 시각화
for date in pd.date_range(start='2018-01-01', end='2023-12-31', freq='ME'):  # 'M' 대신 'ME' 사용
    monthly_data = data[:date]
    returns = monthly_data.pct_change(fill_method=None).dropna()  # fill_method=None 사용
    expected_returns = returns.mean() * 252  # 연간 수익률
    cov_matrix = returns.cov() * 252  # 연간 공분산
    first_date_in_kofr = get_closest_date(date.replace(day=1), df_kofr)
    risk_free_rate = df_kofr.loc[first_date_in_kofr]['KOFR'] * 0.01

    num_portfolios = 10000
    results, weights = generate_random_portfolios(num_portfolios, expected_returns, cov_matrix, risk_free_rate)
    
    # 상위 100개의 Sharpe Ratio를 가진 포트폴리오 선택
    top_100_indices = np.argsort(results[2])[-100:]
    top_100_weights = np.array([weights[i] for i in top_100_indices])
    
    if previous_weights is not None:
        # 코사인 유사도를 사용하여 이전 달의 가중치와 가장 비슷한 포트폴리오 선택
        similarities = cosine_similarity([previous_weights], top_100_weights)
        best_match_idx = np.argmax(similarities)
        best_weights = top_100_weights[best_match_idx]
    else:
        # 이전 달의 가중치가 없는 경우, Sharpe Ratio가 가장 높은 포트폴리오 선택
        best_weights = top_100_weights[np.argmax(results[2][top_100_indices])]
    
    best_weights_record[date.strftime("%Y-%m")] = best_weights
    previous_weights = best_weights
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title(f'Efficient Frontier as of {date.strftime("%Y-%m")}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    image_path = os.path.join(output_dir, f'efficient_frontier_{date.strftime("%Y-%m")}.png')
    plt.savefig(image_path)
    images.append(imageio.imread(image_path))
    plt.close()

# GIF 저장
gif_path = os.path.join(output_dir, 'efficient_frontier.gif')
imageio.mimsave(gif_path, images, duration=0.5)

# Best Portfolio Weights Over Time 그래프 저장
df_weights = pd.DataFrame(best_weights_record).T
df_weights.columns = ['S&P 500', '20+ Year Treasury', 'KOSPI']

plt.figure(figsize=(14, 8))
for asset in df_weights.columns:
    plt.plot(df_weights.index, df_weights[asset], label=asset)

plt.xlabel('Date')
plt.ylabel('Weight')
plt.title('Best Portfolio Weights Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'best_portfolio_weights_over_time.png'))
plt.show()
