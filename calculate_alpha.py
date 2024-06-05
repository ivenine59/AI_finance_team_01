import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


util_a = 10
# Load the data (assumed to be available in the same directory)
data = pd.read_csv('predictions.csv', parse_dates=['date'], index_col='date')

# Define the function for a * sqrt(max(0, x - b)) + c
def model_function(x, a, b, c):
    return a * np.sqrt(np.maximum(0, x - b)) + c

# Derivative of the model function
def model_derivative(x, a, b):
    return np.where(x > b, a / (2 * np.sqrt(x - b)), 0)

# Define the function for y = m*x + risk_free_rate
def linear_function(x, m, risk_free_rate):
    return m * x + risk_free_rate

# Define the quadratic function
def quadratic_function(x, a, risk_free_rate):
    return a * x**2 + risk_free_rate

# Initialize lists to store alpha values
alpha_values = []

# Initialize a figure for plotting
plt.figure(figsize=(14, 10))

# Loop through the first few rows for visualization purposes
for idx, row in data.head(1).iterrows():
    # Define a range of x values
    x = np.linspace(0, 1, 100)
    
    # Model function values
    model_y = model_function(x, row['a'], row['b'], row['c'])
    
    # Find the tangent point where the derivative of the model equals the slope
    def tangent_condition(x):
        return model_derivative(x, row['a'], row['b']) - (row['a'] / (2 * np.sqrt(x - row['b'])))
    
    tangent_x = fsolve(tangent_condition, row['b'] + 0.1)[0]
    tangent_y = model_function(tangent_x, row['a'], row['b'], row['c'])
    
    m = model_derivative(tangent_x, row['a'], row['b'])
    
    # Linear function values
    linear_y = linear_function(x, m, tangent_y - m * tangent_x)
    
    # Quadratic function values
    quadratic_y = quadratic_function(x, util_a, row['risk_free_rate'])
    
    # Find the intersection of the quadratic and linear functions
    def intersection_condition(x):
        return quadratic_function(x, util_a, row['risk_free_rate']) - linear_function(x, m, tangent_y - m * tangent_x)
    
    util_x = fsolve(intersection_condition, 1)[0]
    util_y = quadratic_function(util_x, util_a, row['risk_free_rate'])
    
    # Calculate alpha values
    alpha_x = util_x / tangent_x
    alpha_y = (util_y - row['risk_free_rate']) / (tangent_y - row['risk_free_rate'])
    
    # Store alpha values
    alpha_values.append([idx.date(), alpha_x, alpha_y])
    
    # Plot the model function
    plt.plot(x, model_y, label=f'Model Function (date: {idx.date()})')
    
    # Plot the tangent line
    plt.plot(x, linear_y, linestyle='--', label=f'Tangent Line (date: {idx.date()})')
    
    # Plot the quadratic function
    plt.plot(x, quadratic_y, linestyle='-.', label=f'Quadratic Function (date: {idx.date()})')
    
    # Highlight the tangent point
    plt.plot(tangent_x, tangent_y, 'ro', label=f'Tangent Point (date: {idx.date()})')
    
    # Highlight the intersection point
    plt.plot(util_x, util_y, 'go', label=f'Intersection Point (date: {idx.date()})')

# Save the alpha values to a CSV file
alpha_df = pd.DataFrame(alpha_values, columns=['date', 'alpha_x', 'alpha_y'])
alpha_df.to_csv('alpha_values.csv', index=False)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Function, Tangent Lines, and Quadratic Function')
plt.legend()
plt.grid(True)
plt.savefig('model_function_and_tangent_lines.png')
plt.show()
