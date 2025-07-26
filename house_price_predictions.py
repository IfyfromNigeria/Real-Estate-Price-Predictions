# -*- coding: utf-8 -*-
"""
Spyder Editor

IFUNANYA AKUPUPOME
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('C:/Users/ADMIN/Downloads/Real estate.csv')

def data_summary(df):
    '''Prints a summary of the dataset including info, shape, description
    and missing values'''
    print("Dataset Info:")
    df.info()
    
    print('\nDataset Shape:', df.shape)
    
    print('\nDescriptive Statistics:')
    print(df.describe())
    
    print('\nMissing Values:')
    print(df.isnull().sum())
    
data_summary(data)

# EXPLORARTORY DATA ANALYSIS & VISUALIZATIONS
# CORRELATION HEATMAP
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# PAIRPLOT
sns.pairplot(data[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']])
plt.suptitle('Pairwise Plots of Selected Features', y=1.02)
plt.show()

# HOUSE PRICE DISTRIBUTION
sns.histplot(data['Y house price of unit area'], kde=True, bins=30)
plt.title('Distribution of House Prices per Unit Area')
plt.xlabel('Price')
plt.show()

# PRIVE VS. DISTANCE TO MRT
sns.scatterplot(x='X3 distance to the nearest MRT station', y='Y house price of unit area', data=data)
plt.title('Price vs. Distance to MRT')
plt.xlabel('Distance to MRT')
plt.ylabel('House Price per Unit Area')
plt.show()

# GEOGRAPHICAL DISTRIBUTION OF PRICES (LAT/LONG)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='X6 longitude', y='X5 latitude', hue='Y house price of unit area', palette='viridis')
plt.title('Spatial Distribution of House Prices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# SPLITTING THE DATASET INTO TRAINING AND TESTING
X = data.iloc[:,2:-1]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

# STANDARDIZING THE DATASET
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# APPLYING PCA
pca = PCA(0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# MODEL TRAINING & TESTING
ln = LinearRegression()
ln.fit(X_train_pca, y_train)
y_pred_ln = ln.predict(X_test_pca)

poly = PolynomialFeatures(3)
X_train_poly = poly.fit_transform(X_train_pca)
ln.fit(X_train_poly, y_train)
X_test_poly = poly.transform(X_test_pca)
y_pred_poly = ln.predict(X_test_poly)

knn = KNeighborsRegressor(3)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)

svr = SVR()
svr.fit(X_train_pca, y_train)
y_pred_svr = svr.predict(X_test_pca)

# MODEL EVALUATION
print("Model Performance:\n")

# Linear Regression
print("Linear Regression:")
print("R² score:", r2_score(y_test, y_pred_ln))
print("MSE:", mean_squared_error(y_test, y_pred_ln))
print()

# Polynomial Regression
print("Polynomial Regression (degree 3):")
print("R² score:", r2_score(y_test, y_pred_poly))
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print()

# K-Nearest Neighbors
print("K-Nearest Neighbors (k=3):")
print("R² score:", r2_score(y_test, y_pred_knn))
print("MSE:", mean_squared_error(y_test, y_pred_knn))
print()

# Support Vector Regression
print("Support Vector Regression:")
print("R² score:", r2_score(y_test, y_pred_svr))
print("MSE:", mean_squared_error(y_test, y_pred_svr))
print()