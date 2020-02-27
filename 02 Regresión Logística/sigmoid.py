# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 23:31:26 2020

@author: Daniel
"""

# Importar librería
import matplotlib.pyplot as plt
import numpy as np


# Función Y 

y = np.random.randint(low=0, high=2, size=20)
x = np.arange(0, 20, step = 1)


fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set(xlabel='X', ylabel='Y',
       title='Función Y')
ax.grid()
plt.show()


# Función sigmoidea


# Función sigmoidea
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Visualizar la función sigmoidea
z = np.arange(-10, 10, 0.1)
p_z = sigmoid(z)
plt.plot(z, p_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.title("Función sigmoide")
 # y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()