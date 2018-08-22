#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 07:59:43 2018

@author: andeladeveloper
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the libraries
dataset = pd.read_csv('Salary_Data.csv')

# create a mtrix of the independent variables 
X = dataset.iloc[:, :-1].values
""" 
the first column means all the lines
the second column means all the lines
the column minus one means all the columns minus the last one
"""

# create the dependent variable vector
y = dataset.iloc[:, 1].values # top get the lasst column input it's index in the dataset

# splitting the dataet into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training test results
plt.scatter(X_train, y_train, color='red') # observation point
plt.plot(X_train, regressor.predict(X_train), color='blue') # regression line
plt.title('Salary vs Experience (Training sets)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red') # observation point
plt.plot(X_train, regressor.predict(X_train), color='blue') # regression line
plt.title('Salary vs Experience (Test sets)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


