# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
print('Initial Data: ')
print(dataset)

# Features
X = dataset.iloc[:, :-1].values

# Dependent Variable
y = dataset.iloc[:, -1].values

print('Initial X: ')
print(X)
print('Initial Y: ')
print(y)

# Controlling for missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# [all columns, columns 2 and 3], only choose columns with numerical data
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print('X without missing data: ')
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# One Hot Encoding independent variable of country name
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print('X with hot encoding: ')
print(X)

# Encoding dependent variable (binary, yes or no, 1 or 0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print('Y with label encoding: ')
print(y)

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print('X training set: ')
print(x_train)
print('X testing set: ')
print(x_test)
print('Y training set: ')
print(y_train)
print('Y testing set: ')
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# only scale for numerical data, not the dummy variables like country code
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print('Scaled X training set: ')
print(x_train)
print('Scaled X testing set: ')
print(x_test)