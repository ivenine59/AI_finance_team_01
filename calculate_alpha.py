import pandas as pd
import numpy as np
from scipy.optimize import fsolve

util_a = 0.5
# Load the CSV data into a DataFrame
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

# Define the function for a * x^2 + risk_free_rate
def quadratic_function(x, a, risk_free_rate):
    return a * x**2 + risk_free_rate

# Initialize lists to store the results
tangent_points = []
slopes = []
util_points = []
alpha_x_vals = []
alpha_y_vals = []

# Loop through the rows to fit the models and find intersections
for idx, row in data.iterrows():
    # Define a range of x values
    x = np.linspace(0, 1, 100)
    
    # Find the tangent point where the derivative of the model equals the slope
    def tangent_condition(x):
        return model_derivative(x, row['a'], row['b']) - (row['a'] - row['risk_free_rate']) / (x - row['b'])
    
    tangent_x = fsolve(tangent_condition, row['b'] + 0.1)[0]
    tangent_y = model_function(tangent_x, row['a'], row['b'], row['c'])
    
    m = (tangent_y - row['risk_free_rate']) / tangent_x
    
    # Find the intersection with the quadratic function
    def util_condition(x):
        return linear_function(x, m, row['risk_free_rate']) - quadratic_function(x, util_a, row['risk_free_rate'])
    
    util_x = fsolve(util_condition, 0.5)[0]
    util_y = linear_function(util_x, m, row['risk_free_rate'])
    
    # Calculate alpha_x and alpha_y
    alpha_x = util_x / tangent_x
    alpha_y = (util_y - row['risk_free_rate']) / (tangent_y - row['risk_free_rate'])
    
    # Store the results
    tangent_points.append((tangent_x, tangent_y))
    slopes.append(m)
    util_points.append((util_x, util_y))
    alpha_x_vals.append(alpha_x)
    alpha_y_vals.append(alpha_y)

# Create a DataFrame to store the results
results = pd.DataFrame({
    'date': data.index,
    'tangent_x': [pt[0] for pt in tangent_points],
    'tangent_y': [pt[1] for pt in tangent_points],
    'slope': slopes,
    'util_x': [pt[0] for pt in util_points],
    'util_y': [pt[1] for pt in util_points],
    'alpha_x': alpha_x_vals,
    'alpha_y': alpha_y_vals
})

# Display the DataFrame
print(results.head())

# Save the results to a CSV file
results.to_csv('tangent_points_slopes_and_utils.csv')
