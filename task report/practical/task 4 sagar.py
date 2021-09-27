# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:41:05 2020

@author: abizer
"""

import numpy as np
import plotly.graph_objects as go
import scipy.optimize as optimize
import datetime
eps = 0.001
num_of_point = 1000
init = np.array([0, 0, 0, 1])
min_max = (-10, 10)
bounds = np.array([min_max, min_max, min_max, min_max])
max_iterations = 1000
border_left = -100
border_right = 100
around_num = 3
class Regression:
 def __init__(self, n, description, func, jac=None):
     a, b, d, c = np.random.random(4)
     delta = np.random.randn(n + 1)
     self.x = 3 * np.arange(0, 1 + 1 / n, 1 / n)
     self.y = (1 / (self.x ** 2 - 3 * self.x + 2)).clip(border_left, border_right) + delta
     self.func = func
     self.calls = np.array([0, 0])
     self.description = description
     self.calc = self.calc_operator(lambda x, y, a, b, c, d: (func(x, a, b, c, d) - y) ** 2, 0)
     self.jac = self.calc_operator(jac, 1)
 def calc_operator(self, func, axis):
     def operator(params):
         a, b, c, d = params
         self.calls[axis] += 1
         ans = np.sum(func(self.x, self.y, a, b, c, d), axis=axis)
         return ans
     return operator

def nelder_mead(r):
 return optimize.minimize(r.calc, init, method='Nelder-Mead', options={'xtol': eps}).x
def levenberg_marquardt(r):
 return optimize.least_squares(r.calc, init, jac=r.jac, xtol=eps).x
def simulated_annealing(r):
 return optimize.dual_annealing(r.calc, bounds=bounds, maxiter=max_iterations).x
def differential_evolution(r):
 return optimize.differential_evolution(r.calc, bounds=bounds, maxiter=max_iterations).x
def jac(x, y, a, b, c, d):
 down = x ** 2 + c * x + d # x^2 + cx + d
 up1 = 2 * (a * x + b - y * down) # 2(ax + b - y(x^2 + cx + d)
 part1 = up1 / down ** 2 # 2(ax + b - y(x^2 + cx + d) / (x^2 + cx + d)^2
 part2 = -part1 * (a * x + b) / down # 2(ax + b - y(x^2 + cx + d)(ax + b) / (x^2 + cx + d)^3
 return np.array([x * part1, part1, x * part2, part2])
def time_now():
 date = datetime.datetime.utcnow()
 return date.microsecond + date.second * int(1e6)
def experiment(method, regression):
 start_calls = regression.calls.copy()
 old_time = time_now()
 ans = method(regression)
 calls = regression.calls - start_calls
 '''a, b, c, d, optimize(a, b), f_calls, jac_calls, time'''
 time = time_now() - old_time
 return ans.tolist() + [regression.calc(ans), calls[0], calls[1], time]
def test_all():
 x = 3 * np.arange(0, 1, 1 / num_of_point)
 x_axis = x.tolist()
 rows = []
 for r in regressions:
     fig = go.Figure()
     for m in reg_methods:
         ans = experiment(m[0], r)
         a, b, c, d = ans[0], ans[1], ans[2], ans[3]
         y = r.func(x, a, b, c, d)
         fig.add_trace(go.Scatter(x=x_axis, y=y.tolist(), name=m[1]))
         ans[:5] = np.around(ans, decimals=around_num)[:5]
         rows.append([m[1]] + ans)
     fig.add_trace(go.Scatter(x=x_axis, y=r.y.tolist(), name="data", mode='markers'))
     fig.update_layout(
         xaxis=dict(
             tickmode='linear',
             tick0=0,
             dtick=0.1
        )
      )
     fig.show()
 rows = np.array(rows).transpose()
 table = go.Figure(data=[go.Table(
     header=dict(
         values=['method', 'a', 'b', 'c', 'd', 'D(a, b, c, d)', 'func_calls', 'jac_calls', 'time, ms']),
     cells=dict(values=rows))])
 table.show()
regressions = [
 Regression(num_of_point, "function",
            lambda x, a, b, c, d: (a * x + b) / (x ** 2 + c * x + d),
            jac
 ),
]
reg_methods = [
 [nelder_mead, "nelder mead"],
 [levenberg_marquardt, "Levenberg Marquardt algorithm"],
 [differential_evolution, "differential evolution"],
 [simulated_annealing, "Simulated Annealing"]
]
test_all()