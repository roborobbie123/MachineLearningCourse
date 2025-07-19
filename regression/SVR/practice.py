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

# Predicting a new result and reversing the feature scaling for y
scaler_y.inverse_transform(regressor.predict(scaler_X.transform([[6.5]])).reshape(-1, 1))

# Visualizing SVR results
# reverse feature scaling for x and y
plt.scatter(scaler_X.inverse_transform(X), scaler_y.inverse_transform(y), color = 'red')
plt.plot(scaler_X.inverse_transform(X), scaler_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing SVR results with higher resolution and smoother curve
X_grid = np.arange(min(scaler_X.inverse_transform(X)), max(scaler_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(scaler_X.inverse_transform(X), scaler_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, scaler_y.inverse_transform(regressor.predict(scaler_X.transform(X_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()