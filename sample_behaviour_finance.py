# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:20:33 2022

@author: lixue
"""

import numpy as np
import matplotlib.pyplot as plt

rf = 1.0303
num_1 = 10000
epsilon = np.random.standard_normal(num_1)
consumption_growth = np.exp(0.02+0.02*epsilon)
num_2 = 101
b0 = np.linspace(0,10,num_2)
nvhat = np.zeros(num_1)
x_solution = np.zeros(num_2)

for n in range(num_2):
    x_bi = {"x-":1, "x+":1.1}
    ex = {"x-":np.nan,"x+":np.nan,"x":1}
    while abs(ex["x"]) >= 10**(-4):
        x = (x_bi["x-"]+x_bi["x+"])/2
        x_bi["x"] = x
        for i,j in x_bi.items():
            for m in range(num_1):
                if j*consumption_growth[m]>=rf:
                    nvhat[m] = j*consumption_growth[m]-rf
                else:
                    nvhat[m] = 2*(j*consumption_growth[m]-rf)
            ex[i] = 0.99*b0[n]*np.mean(nvhat)+0.99*j-1
        if ex["x"] < 0:
            x_bi["x-"] = x_bi["x"]
        elif ex["x"] > 0:
            x_bi["x+"] = x_bi["x"]
    else:
        xa = x_bi["x"]
    x_solution[n] = xa

PD_ratio = 1/(x_solution-1)
print(PD_ratio)

fig,ax = plt.subplots(figsize=(10,8))
ax.plot(b0,PD_ratio)
plt.xlabel('b0')
plt.ylabel('Price-dividend Ratio')
plt.title('price-dividend ratio vs b0')

Mkt_rt = x_solution*np.mean(consumption_growth)
Equity_premium = Mkt_rt - 1.0303
fig,ax = plt.subplots(figsize=(10,8))
ax.plot(b0,Equity_premium)
plt.xlabel('b0')
plt.ylabel('Equity Premium')
plt.title('Equity Premium vs b0')