# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:08:29 2022

@author: XuebinLi

"""




"""
Session 8 :Behavioural Finance

Consider a Barberis, Huang and Santos (2001) economy with the 
following parameter choices for the investor's utility function:


Consumption growth has lognormal distribution:

where ε is a standard normal random variable. 
Simulate the distribution for consumption growth with (at least) 104 random draws for ε. 

The risk-free rate is constant at 1.0303 per year. 
Let x be one plus the dividend yield for the market portfolio:
 
and define the error term:


where utility from recent financial gain or loss is given by:

Calculate the equilibrium values of x for b0 in the range from 0 to 10, 
in increments of 0.1 (or less), using bisection search:

Set x– = 1 and x+ = 1.1, and use the simulated distribution 
of consumption growth to confirm that e(x–) < 0 and e(x+) > 0 ⇒ 
equilibrium value of x must lie between x– and x+.

Set x0 = 0.5*(x– + x+), and use the simulated distribution of 
consumption growth to calculate e(x0).

If |e(x0)| < 10–5, then x0 is (close enough to) the equilibrium value of x.

Otherwise, if e(x0) < 0, then the equilibrium value of x 
lies between x0 and x+, so repeat the procedure with x– = x0.

Otherwise, if e(x0) > 0, then the equilibrium value of x lies 
between x– and x0, so repeat the procedure with x+ = x0.
→ Use the equilibrium value of x to calculate the price-dividend ratio for the market portfolio:

 
 
and plot the price-dividend ratio (on the vertical axis) vs b0. 

→ Use the equilibrium value of x to calculate the expected market return:


and plot the equity premium (on the vertical axis) vs b0. 

→ Briefly explain the economic significance of the investor's utility 
function for recent financial gain or loss [ν(R)], as well as the economic significance of b0 and λ.

Economic significance:

Utility function for recent financial gain or loss is based on prospect theory, 
where financial gain or loss is measured relative to reference level based on risk-free rate

Investor is more sensitive to financial loss, and λ determines degree of loss aversion

b0 determines amount of emphasis that investor puts on utility from recent 
financial gain or loss, compared to utility of consumption

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
    ex = {"x-":np.nan,"x+":np.nan,"x0":1.05}
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

#print                          
df_b0 = pd.DataFrame({"b0": b0, "Equilibrium value of x": x_answer})
df_price_dividend_ratio = pd.DataFrame({"b0": b0, "Equilibrium value of x":x_answer, "Price dividend_ratio": 
                                        Price_dividend_ratio})
df_expected_market_return = pd.DataFrame({"b0": b0, "Equilibrium value of x":x_answer, "Expected Market Return": 
                                          Mkt_return})

print(df_b0)
print("")
print(df_price_dividend_ratio)
print("")
print(df_expected_market_return)

    
    
    
    
    
    
    

