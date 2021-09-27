# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 10:19:25 2020

@author: abizer
"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import numpy as np
from decimal import Decimal
#for constant function
def time1(v):
    a=[]
    for i in range(5):
        start_time=time.time()
        con(v)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def con(v):
    '''returns a condtant value C'''
    return"c"
#for summation in array element
def time2(v):
    a=[]
    n=len(v)
    for i in range(5):
        start_time=time.time()
        soa(v,n)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def soa(v, n): 
     if len(v)== 1: 
        return v[0] 
     else: 
        return v[0]+soa(v[1:], n)
#for product of array element    
def time3(v):
    a=[]
    n=len(v)
    for i in range(5):
        start_time=time.time()
        poa(v,n)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def poa(v, n): 
     if len(v)== 1: 
        return v[0] 
     else: 
        return v[0]+soa(v[1:], n)
#polynomial function
def time4(v):
    a=[]
    for i in range(5):
        start_time=time.time()
        poly(v)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def poly(v):
    '''value of polynomial at x=1.5'''
    k=0
    for i in range(len(v)):
        k+=Decimal(v[i])*(Decimal(1.5)**Decimal(i))
    return k
#honors function
def time5(v):
    a=[]
    for i in range(5):
        start_time=time.time()
        hon(v)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def hon(v):
    '''value of polynomial using horner's at x=1.5'''
    k=0
    for i in range(len(v)):
        k=Decimal(v[i])+Decimal(1.5)*Decimal(k)
    return k
#bubble short
def time6(v):
    a=[]
    for i in range(5):
        start_time=time.time()
        bubbleSort(v)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def bubbleSort(v):
    n=len(v)
    for i in range(n-1):
        for j in range(0,n-1-i):
            if v[j] > v[j+1]: 
                v[j], v[j+1] = v[j+1], v[j]
    return v
#for quick sort
def time7(v):
    a=[]
    for i in range(5):
        n = len(v)
        start_time=time.time()
        quickSort(v,0,n-1)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
def partition(arr,low,high):
   i = ( low-1 )
   pivot = arr[high] # pivot element
   for j in range(low , high):
      # If current element is smaller
      if arr[j] <= pivot:
         # increment
         i = i+1
         arr[i],arr[j] = arr[j],arr[i]
   arr[i+1],arr[high] = arr[high],arr[i+1]
   return ( i+1 )
# sort
def quickSort(arr,low,high):
   if low < high:
      # index
      pi = partition(arr,low,high)
      # sort the partitions
      quickSort(arr, low, pi-1)
      quickSort(arr, pi+1, high)
#for timesort
def time8(v):
    a=[]
    for i in range(5):
        start_time=time.time()
        timsort(v)
        a.append(time.time()-start_time)
    a=sum(a)/5
    return a
#defining insertion for using in timsort
def InsertionSort(array):
    '''Sorting the array using Insertion sort'''
    for x in range (1, len(array)):
        for i in range(x, 0, -1):
            if array[i] < array[i - 1]:
                t = array[i]
                array[i] = array[i - 1]
                array[i - 1] = t
            else:
                break
            i = i - 1
    return array
#defining merge for using timsort
def Merge(aArr, bArr):
    '''Merges he given arrays'''
    a = 0
    b = 0
    cArr = []
    while a < len(aArr) and b < len(bArr):
        if aArr[a] <= bArr[b]:
            cArr.append(aArr[a])
            a = a + 1
        elif aArr[a] > bArr[b]:
            cArr.append(bArr[b])
            b = b + 1
    while a < len(aArr):
        cArr.append(aArr[a])
        a = a + 1
    while b < len(bArr):
        cArr.append(bArr[b])
        b = b + 1
    return cArr
#tim sort
def timsort(v):
    '''Sorting the array using Tim sort'''
    for x in range(0, len(v), 64):
        v[x : x + 64] = InsertionSort(v[x : x + 64])
    RUNinc = 64
    while RUNinc < len(v):
        for x in range(0, len(v), 2 * RUNinc):
            v[x : x + 2 * RUNinc] = Merge(v[x : x + RUNinc], v[x + RUNinc: x +
2 * RUNinc])
    RUNinc = RUNinc * 2
    return v
#for matrix miltiplation
def time9(a,b):
    d=[]
    for j in range(5):
        start_time=time.time()
        r=np.matmul(a,b)
        d.append(time.time()-start_time)
    d=sum(d)/5
    return d
#main driver code
n=np.arange(1,2001)    
t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
for i in range(1,2001):
    v=np.random.rand(i)
    a=np.array(np.random.rand(i,i))  
    b=np.array(np.random.rand(i,i)) 
    t1.append(time1(v))
    t2.append(time2(v))
    t3.append(time3(v))
    t4.append(time4(v))
    t5.append(time5(v))
    t6.append(time6(v))
    t7.append(time7(v))
    t8.append(time8(v))
    t9.append(time9(a,b))
    
    
#Curve Fitting function
def func(x, a, b):
    return a*(x**(b))
# Curve fitting function for linearthamtic time
def func2(x, a, b):
    return a *x* np.log(x ) + b
# Initial guess for the parameters
initialGuess = [0.1,0.1]
#x values for the fitted function
xFit=np.arange(1,2001)

#Plot experimental data points of execution time of constant function
plt.plot(n, t1, '-b',label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t1, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r', label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of constant function')
plt.show()

#Plot experimental data points of execution time of sum of elements function
plt.plot(n, t2, '-b',label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t2, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r', label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of sum of elements function')
plt.show()

#Plot experimental data points of execution time of product of elements function
plt.plot(n, t3, '-b', label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t3, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r',label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of product of elements function')
plt.show()

#Plot experimental data points of execution time of polynomial
plt.plot(n, t4, '-b', label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t4, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r', label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of polynomial')
plt.show()

#Plot experimental data points of execution time of polynomial by honors method
plt.plot(n, t5, '-b', label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t5, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r', label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of polynomial by honors method')
plt.show()

#Plot experimental data points of execution time of bubble sorting
plt.plot(n, t6, '-b', label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t6, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r', label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of bubblesort')
plt.show()

#Plot experimental data points of execution time of quick sortig
#Perform the curve-fit
popt, pcov = curve_fit(func2, n, t7)
plt.figure()
plt.plot(n, t7, label="Experimental time")
#Plot the fitted function
plt.plot(n, func2(n, *popt), '-r', label="Execution time")
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of quicksort')
plt.show()

#Plot experimental data points of execution time of tim sorting
#Perform the curve-fit
popt, pcov = curve_fit(func2, n, t8)
plt.figure()
plt.plot(n, t8, label="Experimental time")
#Plot the fitted function
plt.plot(n, func2(n, *popt), '-r', label="Execution time")
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of timsort')
plt.show()

#Plot experimental data points for execution time of matrix multiplication
plt.plot(n, t9, '-b', label='Experimental time')
#Perform the curve-fit
popt, pcov = curve_fit(func, n, t9, initialGuess)
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), '-r', label='Execution time')
plt.legend()
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of matrix multiplication')
plt.show()
  
    