import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Positions are essentially already coded with the Level column, can skip first column
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training Linear Regression model on entire dataset
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X, y)

# Training Polynomial Regression model on entire dataset
from sklearn.preprocessing import PolynomialFeatures
poly_model = PolynomialFeatures(degree = 2)
X_poly = poly_model.fit_transform(X)
lin_model_2 = LinearRegression()
lin_model_2 = fit(X_poly, y)