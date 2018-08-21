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
dataset = pd.read_csv('Data.csv')

# create a mtrix of the independent variables 
X = dataset.iloc[:, :-1].values
""" 
the first column means all the lines
the second column means all the lines
the column minus one means all the columns minus the last one
"""

# create the dependent variable vector
y = dataset.iloc[:, 3].values # top get the lasst column imput it's index in the dataset

# splitting the dataet into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""