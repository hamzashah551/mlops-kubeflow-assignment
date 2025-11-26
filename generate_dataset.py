"""
Script to generate Boston Housing dataset CSV file
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples (Boston Housing has 506 samples)
n_samples = 506

# Create Boston Housing-like dataset
# Feature descriptions:
# CRIM: per capita crime rate by town
# ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS: proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX: nitric oxides concentration (parts per 10 million)
# RM: average number of rooms per dwelling
# AGE: proportion of owner-occupied units built prior to 1940
# DIS: weighted distances to five Boston employment centres
# RAD: index of accessibility to radial highways
# TAX: full-value property-tax rate per $10,000
# PTRATIO: pupil-teacher ratio by town
# B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT: % lower status of the population
# MEDV: Median value of owner-occupied homes in $1000's (TARGET)

data = {
    'CRIM': np.random.exponential(3.61, n_samples),
    'ZN': np.random.choice([0, 12.5, 25, 50, 75, 100], n_samples, p=[0.4, 0.2, 0.15, 0.15, 0.05, 0.05]),
    'INDUS': np.random.gamma(2, 5.5, n_samples),
    'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),
    'NOX': np.random.beta(2, 2, n_samples) * 0.5 + 0.3,
    'RM': np.random.normal(6.28, 0.7, n_samples),
    'AGE': np.random.beta(2, 1, n_samples) * 100,
    'DIS': np.random.gamma(2, 2, n_samples),
    'RAD': np.random.choice(range(1, 25), n_samples),
    'TAX': np.random.normal(408, 168, n_samples),
    'PTRATIO': np.random.normal(18.5, 2.2, n_samples),
    'B': np.random.beta(10, 1, n_samples) * 400,
    'LSTAT': np.random.gamma(2, 6, n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate target variable (MEDV) based on features with some realistic relationships
# More rooms (RM) -> higher price
# Higher crime (CRIM) -> lower price
# Higher LSTAT -> lower price
# River proximity (CHAS) -> higher price

df['MEDV'] = (
    50 +  # Base price
    8 * (df['RM'] - 6) +  # Rooms effect
    -2 * np.log1p(df['CRIM']) +  # Crime effect
    -0.5 * df['LSTAT'] +  # Lower status effect
    5 * df['CHAS'] +  # River effect
    -0.3 * df['AGE'] / 10 +  # Age effect
    np.random.normal(0, 5, n_samples)  # Random noise
)

# Clip values to realistic range
df['MEDV'] = df['MEDV'].clip(5, 50)

# Round numeric columns
for col in df.columns:
    if col in ['CHAS', 'RAD']:
        df[col] = df[col].astype(int)
    else:
        df[col] = df[col].round(4)

# Display info
print("Boston Housing Dataset Generated")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nStatistics:")
print(df.describe())

# Save to CSV
output_path = 'data/raw/boston_housing.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Dataset saved to: {output_path}")
