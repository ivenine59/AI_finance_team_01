import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Read 'coefficients.csv'
coefficients_df = pd.read_csv('coefficients.csv', index_col=0)

print(coefficients_df.head())

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=150, output_size=4):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(coefficients_df[['a', 'b', 'c', 'risk_free_rate']].values)

# Prepare the data
data = torch.tensor(scaled_data, dtype=torch.float32)

model = LSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

# Train the model
epochs = 400
patience = 10  # Early stopping patience
best_loss = np.inf
patience_counter = 0

for i in range(epochs):
    epoch_loss = 0
    for index in range(len(data) - 1):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        input_seq = data[:index+1]  # Sequence from the beginning to the current date
        y_pred = model(input_seq)
        y = data[index+1]  # Target is the next date's data
        single_loss = loss_function(y_pred, y)
        single_loss.backward()
        optimizer.step()
        epoch_loss += single_loss.item()

    scheduler.step(epoch_loss)

    if i % 10 == 1:
        print(f'epoch: {i:3} loss: {epoch_loss / (len(data) - 1):10.8f}')

    # Early stopping
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {i}')
            break

# Make predictions
model.eval()
predictions = []

for index in range(len(data) - 1):
    with torch.no_grad():
        input_seq = data[:index+1]
        y_pred = model(input_seq)
        predictions.append(y_pred.numpy())

predictions = np.array(predictions)

# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

# to dataframe
predictions_df = pd.DataFrame(predictions, columns=['a', 'b', 'c', 'risk_free_rate'])
predictions_df.index = coefficients_df.index[1:]

# to csv
predictions_df.to_csv('predictions.csv')

# Plot the results
plt.figure(figsize=(14, 8))
# Function to set x-axis ticks to show labels only once a year
def set_yearly_ticks(ax, dates):
    years = pd.to_datetime(dates).year
    unique_years = np.unique(years)
    ax.set_xticks([np.where(years == year)[0][0] for year in unique_years])
    ax.set_xticklabels(unique_years)

# Plot 'a'
ax1 = plt.subplot(2, 2, 1)
ax1.plot(coefficients_df.index[1:], coefficients_df['a'][1:], label='Actual a')
ax1.plot(coefficients_df.index[1:], predictions_df['a'], label='Predicted a')
ax1.legend()
ax1.set_title('Actual vs Predicted a')
set_yearly_ticks(ax1, coefficients_df.index[1:])

# Plot 'b'
ax2 = plt.subplot(2, 2, 2)
ax2.plot(coefficients_df.index[1:], coefficients_df['b'][1:], label='Actual b')
ax2.plot(coefficients_df.index[1:], predictions_df['b'], label='Predicted b')
ax2.legend()
ax2.set_title('Actual vs Predicted b')
set_yearly_ticks(ax2, coefficients_df.index[1:])

# Plot 'c'
ax3 = plt.subplot(2, 2, 3)
ax3.plot(coefficients_df.index[1:], coefficients_df['c'][1:], label='Actual c')
ax3.plot(coefficients_df.index[1:], predictions_df['c'], label='Predicted c')
ax3.legend()
ax3.set_title('Actual vs Predicted c')
set_yearly_ticks(ax3, coefficients_df.index[1:])

# Plot 'risk_free_rate'
ax4 = plt.subplot(2, 2, 4)
ax4.plot(coefficients_df.index[1:], coefficients_df['risk_free_rate'][1:], label='Actual risk_free_rate')
ax4.plot(coefficients_df.index[1:], predictions_df['risk_free_rate'], label='Predicted risk_free_rate')
ax4.legend()
ax4.set_title('Actual vs Predicted risk_free_rate')
set_yearly_ticks(ax4, coefficients_df.index[1:])

plt.tight_layout()

# Save the plot
plt.savefig('predictions.png')