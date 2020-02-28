# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:58:50 2020

@author: Daniel
"""

# Importar librerías

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Algoritmo de gradient descent
class LogisticRegressionGD(object):
    
    def __init__(self, l_rate = 0.1, n_iter =10000):
        self.l_rate = l_rate
        self.n_iter = n_iter               
        
    def fit(self, X, y, theta):
        self.theta = theta
        X_value = X[:,1].reshape(-1, 1)
        const = self.l_rate     
        for i in range(0, self.n_iter):
            h = self.sigmoid(X_value)           
            self.theta[0] = self.theta[0]-const*sum(h-y)
            self.theta[1] = self.theta[1]-const*sum((h-y).transpose().dot(X_value))            
        return self.theta
    
    def sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):        
        value = X.dot(theta)
        predict_value  = 1/(1+np.exp(value))
        for i in predict_value:
            i
        return predict_value  

# Importar el dataset de training
dataset = pd.read_csv('train.csv')
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
X = dataset.iloc[:len(dataset),[5,9]].values
X_train = np.insert(X, 0, 1, axis = 1)
y_train = dataset.iloc[:len(dataset), 1].values.reshape(-1,1)

# Importar el dataset de test
dataset_test = pd.read_csv('test.csv')
dataset_test['Age'] = dataset_test['Age'].fillna(dataset_test['Age'].mean())
dataset_test['Fare'] = dataset_test['Fare'].fillna(dataset_test['Fare'].mean())
X_test = dataset_test.iloc[:len(dataset_test), [4,8]].values
X_test = np.insert(X_test, 0, 1, axis = 1)

# Aplicación del modelo con gradient descent
regression = LogisticRegressionGD(l_rate = 0.01, n_iter = 100)
coef = regression.fit(X_train, y_train, np.array([0.1, 0.1]))
y_predict = regression.predict(X_test)


# Aplicación del modelo con librería de sklearn
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state = 0)
logistic.fit(X_train, y_train)
y_predict_py = logistic.predict(X_test)


# Visualización de los datos
dataset_train = dataset['Survived'==0]
plt.scatter(X[:,0],X[:,1], color = "red")
plt.xlabel("Edad de los pasajeros")
plt.ylabel("Tarifa del ticket")
plt.title("Edad vs Tarifa del TITANIC (training data)")
plt.show()



# Visualización del dataset
group_data = dataset.groupby(['Survived', 'Pclass']).count()
labels = ['1°', '2°', '3°']
survived = group_data.iloc[3:6, 1].values
dead = group_data.iloc[0:3,1].values
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, survived, width, label='Sobreviviente')
rects2 = ax.bar(x + width/2, dead, width, label='Fallecido', color = "red")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('N° personas')
ax.set_title('Sobrevivientes y Fallecidos del TITANIC por Clase (Training Data)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.ylim(0, 400)
plt.xlabel('Clase')
plt.show()



