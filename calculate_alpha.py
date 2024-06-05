import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Load the data (assumed to be available in the same directory)
data = pd.read_csv('predictions.csv', parse_dates=['date'], index_col='date')

# Define the function for a * sqrt(max(0, x - b)) + c
def model_function(x, a, b, c):
    return a * np.sqrt(np.maximum(0, x - b)) + c

# Derivative of the model function
def model_derivative(x, a, b):
    return a / (2 * np.sqrt(np.maximum(0, x - b)))

# Define the function for y = m*x + risk_free_rate
def linear_function(x, m, risk_free_rate):
    return m * x + risk_free_rate

# Initialize a figure for plotting
plt.figure(figsize=(14, 10))

# Loop through the first few rows for visualization purposes
for idx, row in data.head(5).iterrows():
    # Define a range of x values
    x = np.linspace(0, 1, 100)
    
    # Model function values
    model_y = model_function(x, row['a'], row['b'], row['c'])
    
    # Find the tangent point where the derivative of the model equals the slope
    def tangent_condition(x):
        return model_derivative(x, row['a'], row['b']) - (row['a'] - row['risk_free_rate']) / (x - row['b'])
    
    tangent_x = fsolve(tangent_condition, row['b'] + 0.1)[0]
    tangent_y = model_function(tangent_x, row['a'], row['b'], row['c'])
    
    m = (tangent_y - row['risk_free_rate']) / tangent_x
    
    # Linear function values
    linear_y = linear_function(x, m, row['risk_free_rate'])
    
    # Plot the model function
    plt.plot(x, model_y, label=f'Model Function (date: {idx.date()})')
    
    # Plot the tangent line
    plt.plot(x, linear_y, linestyle='--', label=f'Tangent Line (date: {idx.date()})')
    
    # Highlight the tangent point
    plt.plot(tangent_x, tangent_y, 'ro', label=f'Tangent Point (date: {idx.date()})')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Function and Tangent Lines')
plt.legend()
plt.grid(True)
plt.show()
