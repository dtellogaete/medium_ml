# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:41:52 2020

@author: Daniel
"""

# Importar las librerías 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Algoritmo de Descenso de Gradiente 
class LinearRegressionGD(object):
    
    def __init__(self, l_rate = 0.1, n_iter =10000):
        self.l_rate = l_rate
        self.n_iter = n_iter               
        
    def fit(self, X, y, theta):
        self.theta = theta
        X_value = X[:,1].reshape(-1, 1)
        const = self.l_rate*(1/X.shape[0])        
        for i in range(0, self.n_iter):
            h = X.dot(self.theta)            
            self.theta[0] = self.theta[0]-const*sum(h-y)
            self.theta[1] = self.theta[1]-const*sum((h-y).transpose().dot(X_value))            
        return self.theta
    
    def predict(self, X):
        X_test = X[:, 1]
        predict_value = X_test*self.theta[1]+self.theta[0]
        return predict_value   
       
# Importar el dataset de entrenamiento
dataset = pd.read_csv('Admission_Predict_Ver1.1.csv')
X = dataset.iloc[:len(dataset), 1].values
X = X.reshape(-1,1)
X = np.insert(X, 0, 1, axis = 1)
y = dataset.iloc[:len(dataset), -1].values.reshape(-1,1)

# Seleccionar conjunto de training y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 0)

# Escalado de las variables
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)
st_y = StandardScaler()
y_train = st_y.fit_transform(y_train).reshape(-1)
y_test = st_y.transform(y_test).reshape(-1)

# Aplicación del modelo con grad
regression = LinearRegressionGD(l_rate = 0.01, n_iter = 20000)
coef = regression.fit(X_train, y_train, np.array([0.0, 1]))
y_predict = regression.predict(X_test)

# Aplicación del modelo con librería de sklearn
from sklearn.linear_model import LinearRegression
regression_py = LinearRegression() 
regression_py.fit(X_train, y_train)
y_predict_py = regression_py.predict(X_test) 

# Gráfica de regresión conjunto de test
plt.scatter(X_test[:,1], y_test, color = "red") 
plt.plot(X_test[:,1], y_predict_py, color = "blue")
plt.legend(('Descenso de Gradiente',),
            loc='lower right')
plt.title("Probabilidad admisión vs GRE Score (Conjunto de Test)")
plt.xlabel("GRE Score")
plt.ylabel("Probabilidad admisión a Postgrado")
plt.show()
