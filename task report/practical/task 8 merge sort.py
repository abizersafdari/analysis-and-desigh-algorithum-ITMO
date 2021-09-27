# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:01:22 2020

@author: abizer
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import numpy as np
def time1(v):
    a=[]
    for i in range(5):
        start_time=time.time()
        mergeSort(array)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a

#merge sort using divide and conqure
def mergeSort(array):
    if len(array) > 1:

        #  r is the point where the array is divided into two subarrays
        r = len(array)//2
        L = array[:r]
        M = array[r:]

        # Sort the two halves
        mergeSort(L)
        mergeSort(M)

        i = j = k = 0

        # Until we reach either end of either L or M, pick larger among
        # elements L and M and place them in the correct position at A[p..r]
        while i < len(L) and j < len(M):
            if L[i] < M[j]:
                array[k] = L[i]
                i += 1
            else:
                array[k] = M[j]
                j += 1
            k += 1

        # When we run out of elements in either L or M,
        # pick up the remaining elements and put in A[p..r]
        while i < len(L):
            array[k] = L[i]
            i += 1
            k += 1

        while j < len(M):
            array[k] = M[j]
            j += 1
            k += 1


# Print the array
def printList(array):
    for i in range(len(array)):
        print(array[i], end=" ")
    print()


# Driver program
n=np.arange(1,1001)
t1=[]
for i in range(1,1001):
    array=[]
    for i in range(100):
        array.append(np.random.randint(100))
    t1.append(time1(array))


#Curve Fitting function
def func(x, a, b):
 return a*(x**(b))
#Curve fitting function for linearthamtic time
def func2(x, a, b):
 return a *x* np.log(x ) + b
# Initial guess for the parameters
initialGuess = [0.1,0.1]
#x values for the fitted function
xFit=np.arange(1,1001)
#Plot experimental data points of execution time of quick sortig
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t1)
plt.figure()
plt.plot(n, t1, label="Experimental time")
#Plot the fitted function
plt.plot(n, func2(n, *popt), '-r', label="Execution time")
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of merge sort')
plt.show()


