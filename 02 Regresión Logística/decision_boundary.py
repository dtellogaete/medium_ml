# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:07:43 2020

@author: Daniel
"""

# Importar librería
import matplotlib.pyplot as plt
import numpy as np

# Decisión Boundary
y_zero = []
x_zero = []
i = 0
while len(y_zero)<50:
    ran_value_y = np.random.rand(1)*10
    ran_value_x = np.random.rand(1)*10   
    if (ran_value_y[0] +ran_value_x[0] -8) < 0:          
        y_zero.append(ran_value_y[0])
        x_zero.append(ran_value_x[0])
        
y_one = []
x_one = []
i = 0
while len(y_one)<50:
    ran_value_y = np.random.rand(1)*10
    ran_value_x = np.random.rand(1)*10   
    if (ran_value_y[0] +ran_value_x[0] -8) >= 0: 
        y_one.append(ran_value_y[0])
        x_one.append(ran_value_x[0])
     

# Función lineal
def lineal(x):
    return -(x-8)

# Visualizar decision bound
x = np.arange(0, 10, 0.2)
plt.plot(x, lineal(x), color= "red", label ="y = -x+8")
plt.scatter(x_zero, y_zero, label = "y=0")
plt.scatter(x_one, y_one, color="green", label ="y=1")
plt.ylim(-0.1, 1.1)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Decision Boundary")
 # y axis ticks and gridline
plt.yticks(np.arange(0, 10, 2))
ax = plt.gca()
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, 
           loc = 'upper right' )
plt.tight_layout()
plt.show()