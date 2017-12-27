# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:55:18 2017

@author: paulo
"""
CROSSOVER_RATE = 0.9 #RATE OF CROSSOVER
MUTATION_RATE = 0.001 #RATE OF MUTATION
POP_SIZE = 100 #POPULATION SIZE
N_GENERATIONS = 1000 #MAXIMUM NUMBER OF GENERATIONS
CHRM_SIZE = 64 #CHROMOSSOME SIZE IN BITS

import numpy as np

#creates a new population
def NewPop():
    return np.random.randint(0, 2, size=(POP_SIZE, CHRM_SIZE))

#converts a binary number represented in a array into an integer number
def BinToInt(nBin):
    return int(''.join(str(x) for x in nBin), 2)
#function to be maximized
def Fitness(x):
    return 1/(1+f(x))
#function f(x)
def f(x):
    return -20*np.exp(-0.2*np.sqrt((1.0/x.__len__())*sum([xi**2 for xi in x]))) - np.exp((1.0/x.__len__())*sum([np.cos(2*np.pi*xi) for xi in x])) + 20 + np.exp(1)

'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-100.0, 100.0, 0.1)
y = x[:]
x, y = np.meshgrid(x, y)
z = x[:]+5
for i in range(0,x.__len__()):
    for j in range(0,x.__len__()):
        z[i][j] = Fitness([x[i][j],y[i][j]])
print z
print x
print y

ax.plot_surface(x, y, z, cmap=cm.coolwarm)
'''