#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:26:23 2018

@author: Bernabé Gonzalez García bernabegoga@gmail.com
"""
# POLYNOMIAL REGRESSION

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = 2)
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualising the Linear Regression results
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color ="blue")

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = "orange")
plt.plot(X, lin_reg_2.predict(X_poly), color ="purple")

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))