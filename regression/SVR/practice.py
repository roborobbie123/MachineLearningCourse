import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# Format needed for the StandardScaler class, a 2D array
y = y.reshape(len(y), 1)
print('Standard Data:')
print(X)
print(y)


# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)
print('Feature Scaled Data: ')
print(X)
print(y)

# Training SVR model on whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
