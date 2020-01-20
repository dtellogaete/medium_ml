# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:41:52 2020

@author: Daniel
"""

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
        for i in range(0, self.n_iter):
            h = X.dot(self.theta)
            const = self.l_rate*(1/X.shape[0])
            self.theta[0] = self.theta[0]-const*sum(h-y)
            self.theta[1] = self.theta[1]-const*sum((h-y).transpose().dot(X_value))            
        return self.theta
    
    def predict(self, X):
        X_test = X[:, 1]
        predict_value = X_test*self.theta[1]+self.theta[0]
        return predict_value   

# Importar el dataset
dataset = pd.read_csv('Consumo_cerveja.csv')
X = list(dataset.iloc[:365, 2].values)
X = np.asarray([float(a.replace(",",".")) for a in X])
X = X.reshape(-1,1)
X = np.insert(X, 0, 1, axis = 1)
y = dataset.iloc[:365, 6].values

# Seleccionar conjunto de training y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, 
                                                    random_state = 0)   
   
# Aplicación del modelo
regression = LinearRegressionGD(l_rate = 0.006, n_iter = 100000)
coef = regression.fit(X_train, y_train, np.array([20, 0.6]))
y_predict = regression.predict(X_test)

# Aplicación del modelo con librería de sklearn
from sklearn.linear_model import LinearRegression
regression_py = LinearRegression() 
regression_py.fit(X_train, y_train)
y_predict_py = regression_py.predict(X_test)

# Gráfica de regresión conjunto de test
plt.scatter(X_test[:,1], y_test, color = "red") 
plt.plot(X_test[:,1], y_predict, color = "blue") 
plt.legend(('Descenso de Gradiente',),
           loc='lower right')
plt.title("Consumo de cerveza vs Temperatura Sao Paulo (Conjunto de Test)")
plt.xlabel("Temperatura °C Sao Paulo")
plt.ylabel("Consumo cerveza en Litros")
plt.show()




        