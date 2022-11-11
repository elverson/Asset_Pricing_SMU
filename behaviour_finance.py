# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:08:29 2022

@author: XuebinLi

"""
"""
Session 8 :Behavioural Finance


"""

import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


"""
e = create 10,000 random draws of standard normal
put this 10,000 into consumption formula(g = np.exp(0.02+0.02*e))
b0 = use linspace to create 101 values from 0 to 1
v = 10,000 zeros
x_answer = 101 zeros
set sigma, gamma and lamda as 0.99, 1, 1
"""


random_draws = 10000
intervals = 101
sigma, gamma, lamda = 0.99, 1, 2
rf_rate = 1.0303
e = np.random.standard_normal(random_draws)
g = np.exp(0.02+0.02*e)
print(g.shape)
b0 = np.linspace(0,10,intervals)
v = np.zeros(random_draws)
x_answer = np.zeros(intervals)
xa = 0


"""
1st 
for-loop x from 0 to 101

set x- as nan, x+ as nan and x0 as 1.05, which means for every 
values of for-loop x. This value will reset at nan,nan and 1.05

    2nd
    while loop. stop loop if value < 0.00001
    set mid value as [positive value + negative value] divide by 2
    
        3rd
        for loop i and j in dictionary_x items
        i refers to x-,x+ and x0
        j refers to np.nan, np.nan and 1.05. of cause these values will chance

            4th
            for loop values in consumption growth g. from index k starting
            with 0 to 10000. If g[k] >= riskfree rate then set v = r - 1.0303
            if not set v = 2(r-1.0303)
            set sigma*b0[x]*np.mean(v)+sigma*j*np.mean(np.power(g,1-gamma))-gamma to each value of ex[i]
            if the value of ex[i] is more than 0 then change positive value to mid. If not change negative value to mid.
            Lastly, it will go back to while loop and check if absolute mid value is smaller than 0.00001. if it is not. Then continue looping



For every value of x. Total of 101. Starting mid value is 1.05.
while the absolute mid value is bigger than 00001 continue loop, if not stop looping. 
We set mid value = positive value + negative value divide by 2
Again we loop through 10,000 values of random draws. for each value get the consumption result and take the mean of the 10,000 results. 
if mid value is bigger than rf_rate we use 1 time consumption formula but if it is lesser than riskfree rate we use 2time consumption value
if the mean value of consumption is bigger than 0 we set positive value as mid value but if consumption is lesser than 0 we set negative value as mid value.
Then if it absoulte value of mid less than 00001 stop looping. if not keep looping until the absolute value is less than 00001.
After the value is found. we go to 2nd value of x. until we have 101 values.



"""

#1st
for x in range(intervals):
    dictionary_x = {"x-":1,"x+":1.1}
    ex = {"x-":np.nan,"x+":np.nan,"x0":1.05}
    
    #2nd
    while abs(ex["x0"]) >= 10**(-5):
        dictionary_x["x0"] = (dictionary_x["x+"]+dictionary_x["x-"])*0.5
        
        #3rd
        for i,j in dictionary_x.items():
            
            #4th
            for k in range(random_draws):
                if j*g[k] >=rf_rate:
                    v[k] = j*g[k]-rf_rate
                else:
                    v[k] = 2*(j*g[k]-rf_rate)
            #ex[i] = sigma*b0[x]*np.mean(v)+sigma*j-gamma
            #page 10
            ex[i] = sigma*b0[x]*np.mean(v)+sigma*j*np.mean(np.power(g,1-gamma))-gamma
        
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

    
    
    
    
    
    
    

