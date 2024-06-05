import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Read 'coefficients.csv'
coefficients_df = pd.read_csv('coefficients.csv', index_col=0)

print(coefficients_df.head())

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=4):
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

model = LSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare the data
data = coefficients_df[['a', 'b', 'c', 'risk_free_rate']].values
data = torch.tensor(data, dtype=torch.float32)

# Train the model
epochs = 150

for i in range(epochs):
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

    if i % 10 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')