import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
    """
    Calculate portfolio return and standard deviation.
    """
    portfolio_return = np.dot(weights, returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_stddev

def generate_random_portfolios(num_portfolios, returns, cov_matrix):
    """
    Generate random portfolios with given returns and covariance matrix.
    """
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_stddev = portfolio_performance(weights, returns, cov_matrix)
        
        results[0,i] = portfolio_stddev
        results[1,i] = portfolio_return
        results[2,i] = results[1,i] / results[0,i] # Sharpe ratio
    
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

# Function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# LSTM model prediction for each asset
predicted_returns = []

for asset in assets:
    dataset = returns[asset].values.reshape(-1, 1)
    
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # Split into train and test sets
    training_size = int(len(dataset) * 0.75)
    test_size = len(dataset) - training_size
    train_data, test_data = dataset[0:training_size, :], dataset[training_size:len(dataset), :1]
    
    # Create datasets
    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float().unsqueeze(-1)
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float().unsqueeze(-1)
    y_test = torch.from_numpy(y_test).float()
    
    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
    
    # Initialize the model, loss function and optimizer
    model = LSTMModel()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    epochs = 100
    print(f'Training model for {asset}')
    for epoch in tqdm(range(epochs)):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, seq.size(0), model.hidden_layer_size),
                                 torch.zeros(1, seq.size(0), model.hidden_layer_size))
            
            y_pred = model(seq)
            
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {single_loss.item()}')
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, X_test.size(0), model.hidden_layer_size),
                             torch.zeros(1, X_test.size(0), model.hidden_layer_size))
        test_predict = model(X_test)
        test_predict = scaler.inverse_transform(test_predict.numpy().reshape(-1, 1))
    
    # Calculate predicted returns
    predicted_return = test_predict.mean() * 252  # Annualized predicted return
    predicted_returns.append(predicted_return)

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