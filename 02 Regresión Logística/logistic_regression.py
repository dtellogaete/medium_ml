# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:58:50 2020

@author: Daniel
"""

# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Algoritmo de gradient descent
class LogisticRegressionGD(object):
    
    def __init__(self, l_rate = 0.1, n_iter =10000, random_state =1):
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.random_state = random_state               
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)        
        self.theta = rgen.normal(loc = 0.0, scale = 0.01,
                                 size = 1 + X.shape[1])     
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            h = self.sigmoid(net_input)   
            errors = y-h
            self.theta[1:] += -self.l_rate*X.T.dot(errors) 
            self.theta[0] += -self.l_rate*errors.sum()          
        return self.theta
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def net_input(self, X):
        return np.dot(X, self.theta[1:]) + self.theta[0]
    
    def predict(self, X):        
        return np.where(self.sigmoid(self.net_input(X))>= 0.5, 0, 1)

# Importar el dataset de training
dataset = pd.read_csv('Social_Network_Ads.csv')
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['EstimatedSalary'] = dataset['EstimatedSalary'].fillna(dataset['EstimatedSalary'].mean())
X = dataset.iloc[:len(dataset),[2,3]].values
y = dataset.iloc[:len(dataset), -1]

# Dividir dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Aplicación del modelo con gradient descent
regression = LogisticRegressionGD(l_rate = 0.0000001, n_iter = 20000)
coef = regression.fit(X_train, y_train)
y_predict = regression.predict(X_test)

# Aplicación del modelo con librería de sklearn
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state = 0)
logistic.fit(X_train, y_train)
y_predict_py = logistic.predict(X_test)

# Matriz de confusión para ver resultados finales
from sklearn.metrics import confusion_matrix
cm_sklearn = confusion_matrix(y_test, y_predict_py)
cm_GD = confusion_matrix(y_test, y_predict)

# Visualización del dataset

dataset_0 = dataset[dataset['Purchased']==0]
dataset_1 = dataset[dataset['Purchased']==1]
X_0 = dataset_0.iloc[:len(dataset_0),[2,3]].values
X_1 = dataset_1.iloc[:len(dataset_1),[2,3]].values
plt.scatter(X_0[:,0],X_0[:,1], color = "red", label = "Purchased = 0")
plt.scatter(X_1[:,0],X_1[:,1], color = "blue", label = "Purchased = 1")
plt.xlabel("Edad de los usuarios")
plt.ylabel("Salario Anual $")
plt.legend(loc = 'upper right', shadow = True)
plt.title("Edad vs Salario")
plt.show()

# Visualización de resultados conjunto de test
def lineal_sklearn(x):
    return (-logistic.coef_[0][0]*x-logistic.coef_[0][1])

def lineal_GD(x):
    return coef[2]*x+coef[1]

def lineal_glmR(x):
    return -2.616*x-1.4

data_vix = pd.DataFrame({'Age':X_test[:,0], 'Salary': X_test[:,1], 
                        'Purchased': y_test})
data_vix_0 = data_vix[data_vix['Purchased']==0]
data_vix_1 = data_vix[data_vix['Purchased']==1]
X_0 = data_vix_0.iloc[:len(data_vix_0),[0,1]].values
X_1 = data_vix_1.iloc[:len(data_vix_1),[0,1]].values
plt.scatter(X_0[:,0],X_0[:,1], color = "red", label = "Purchased = 0")
plt.scatter(X_1[:,0],X_1[:,1], color = "blue", label = "Purchased = 1")
X = data_vix.iloc[:len(data_vix_0),0].values
plt.plot(X, lineal_sklearn(X), color = "darkslategray", label = "sklearn")
plt.plot(X, lineal_GD(X), color = "green", label = "GD")
plt.plot(X, lineal_glmR(X), color = "orange", label = "R")
plt.ylim(-3.5, 3.5)
plt.xlabel("Edad de los usuarios")
plt.ylabel("Salario Anual $")
plt.legend(loc = 'upper right', shadow = True)
plt.title("Edad vs Salario")
plt.show()


