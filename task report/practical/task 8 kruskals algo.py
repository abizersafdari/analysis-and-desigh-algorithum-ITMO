# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:01:22 2020

@author: abizer
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import numpy as np
def a_matrix(x):
    ''' generates a adjacancy matrix of size x'''
    g=np.zeros([x+1, x+1])
    g[:, 0]= np.arange(x+1)
    g[0, :]=np.arange(x+1)
    for i in range(1,x+1):
        for j in range(1,x+1):
            if i !=j:
                g[i, j]=g[j, i]=np.random.randint(1,10)
            if sum(sum(g[1:, 1:]))>=9000:
                break
        if sum(sum(g[1:,1:]))>=9000:
            break
    return g
def time1(v):
    a=[]
    g= Graph(len(v))
    for i in range(len(v)):
        for j in range (i,len(v)):
            c=v[i][j]
            if c!=0:
              g.add_edge(i, j, c)  
    for i in range(5):
        start_time=time.time()
        g.kruskal_algo()
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a

# Kruskal's algorithm in Python


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print("%d - %d: %d" % (u, v, weight))


# Driver program
n=np.arange(1,101)
t1=[]
for i in range(1,101):
    f=a_matrix(i)
    t1.append(time1(f))


#Curve Fitting function
def func(x, a, b):
 return a*(x**(b))
#Curve fitting function for linearthamtic time
def func2(x, a, b):
 return a *x* np.log(x ) + b
# Initial guess for the parameters
initialGuess = [0.1,0.1]
#x values for the fitted function
xFit=np.arange(1,101)
#Plot experimental data points of execution time of quick sortig
#Perform the curve-fit
popt, pcov = curve_fit(func2, n, t1)
plt.figure()
plt.plot(n, t1, label="Experimental time")
#Plot the fitted function
plt.plot(n, func2(n, *popt), '-r', label="Execution time")
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of kruskals algorithm')
plt.show()


