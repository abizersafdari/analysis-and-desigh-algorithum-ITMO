# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:35:50 2020

@author: abizer
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
# generating noise data
xk=np.arange(0, 3.003, 0.003)
fk=1/(xk**2-3*xk+2)
yk=np.zeros(1001)
for i in range(1001):
    if fk[i]< -100:
        yk[i]=-100+np.random.normal(0,1)
    elif fk[i]> 100:
        yk[i]=100+np.random.normal(0,1)
    else:
        yk[i]=fk[i]+np.random.normal(0,1)

# defining rational approximatin function.
def fun(x):
    fun=(x[0]*xk+x[1])/(xk**2+x[2]*xk+x[3])
    return fun

# defining least squares of rational approximation function
def D(x):
    return sum((fun(x)-yk)**2)

# defining residual function
def residual(x):
    return fun(x)-yk

#defining jacobian matrix for rational approximation function
def jacobian(x):
    j=np.empty((xk.size, x.size))
    den=(xk**2+x[2]*xk+x[3])
    j[:,0]=xk/den
    j[:,1]=1/den
    j[:,2]=-xk*fun(x)/den
    j[:,3]=-fun(x)/den
    return j

# intial approximations
x0=np.array([0.5,0.5,1,1])

# caluclating Rational approximation using Nelder mead.
res_nm=optimize.minimize(D, x0, method='Nelder-Mead',\
                         options={'maxiter':1000, 'disp':True, 'fatol':0.001} )

    #caluclating rational approximation using evenberg-Marquardt algorithm
res_lm=optimize.least_squares(residual, x0, jac=jacobian,\
                              method='lm',ftol=0.001 )

# caluclating rational apprximation using simulated anneling.
lw = [-10] * 4
up = [10] * 4
bounds=list(zip(lw, up))
res_sm = optimize.dual_annealing(D, bounds,maxiter=4, accept=1, \
 seed=1234 )

# caluclating rational approximation using differential evalution.
res_de=optimize.differential_evolution(D, bounds, maxiter=1000, atol=0.001)

# Caluclating rational approximation using particle swarn optimizaion
bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)] # upper and lower bounds of variables
nv = 4 # number of variables

y_nm=fun(res_nm.x)
y_lm=fun(res_lm.x)
y_sm=fun(res_sm.x)
y_de=fun(res_de.x)

# plotting approximation function
plt.plot(xk, yk,'bo' ,label='Noisy data')
plt.plot(xk, y_sm, '-g', label='Simulated annealing')
plt.plot(xk, y_lm, '-r', label='Levenberg-Marquardt')
plt.plot(xk, y_nm, '-y' , label='Nelder-mead')
plt.plot(xk, y_de, '-k', label='Didderential Evolution')
plt.title('Rational approximation of noisy data')
plt.ylabel('Noisy data')
plt.xlabel('x-axis')
plt.legend()
plt.show()
