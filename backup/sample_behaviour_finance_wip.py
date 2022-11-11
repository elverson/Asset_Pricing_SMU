# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:08:29 2022

@author: XuebinLi
"""

import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

random_draws = 10000
intervals = 101
sigma, gamma, lamda = 0.99, 1, 2
rf_rate = 1.0303
e = np.random.standard_normal(random_draws)
g = np.exp(0.02+0.02*e)
b0 = np.linspace(0,10,intervals)
v = np.zeros(random_draws)
x_answer = np.zeros(intervals)
xa = 0

for x in range(intervals):
    dictionary_x = {"x-":1,"x+":1.1}
    ex = {"x-":np.nan,"x+":np.nan,"x0":1}
    while abs(ex["x0"]) >= 10**(-5):
        dictionary_x["x0"] = (dictionary_x["x+"]+dictionary_x["x-"])*0.5
        for i,j in dictionary_x.items():
            for k in range(random_draws):
                if j*g[k] >=rf_rate:
                    v[k] = j*g[k]-rf_rate
                else:
                    v[k] = 2*(j*g[k]-rf_rate)
            ex[i] = sigma*b0[x]*np.mean(v)+sigma*j-gamma
        
        if ex[i]>0:
            dictionary_x["x+"] = dictionary_x["x0"]
        elif ex[i]<0:
            dictionary_x["x-"] = dictionary_x["x0"]
            
    else:
        xa = dictionary_x["x0"]
    x_answer[x] = xa
            
Price_dividend_ratio = 1/(x_answer-1)
print(Price_dividend_ratio)
            
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(b0,Price_dividend_ratio)
plt.xlabel('b0')
plt.ylabel('Price-dividend Ratio')
plt.title('price-dividend ratio vs b0')

Mkt_return = x_answer*np.mean(g)
Equity_premium = Mkt_return - rf_rate
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(b0,Equity_premium)
plt.xlabel('b0')
plt.ylabel('Equity Premium')
plt.title('Equity Premium vs b0')

            
                



