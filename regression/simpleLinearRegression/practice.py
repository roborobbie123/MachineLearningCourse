# Import libraries
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

# Import data
dataset = pd.read_csv('Salary_Data.csv')
print(dataset)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training simple linear regression model on dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting test set results, use X_test to predict y
y_pred = regressor.predict(X_test)
print(y_pred)
print(y_test)

# Visualizing training set results
plt.scatter(X_train, y_train, color = 'red')

# predicting the y values of the X_train values for the regression line
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing test set results
plt.scatter(X_test, y_test, color = 'red')

# regression line will be the same
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Prediction for a single value
print(f'Salary prediction for 12 YOE: {regressor.predict([[12]])}')

# Coefficients for linear regression formula
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')