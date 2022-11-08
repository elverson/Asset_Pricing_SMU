# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:51:47 2022

@author: lixue
"""

import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def monte_carlo_weight(rows,columns):
    e = np.random.standard_normal(size=(rows, columns))
    df = pd.DataFrame(data=e)
    return df

def ln_g(dataframe_monte_carlo_weight): 
    df = dataframe_monte_carlo_weight
    df['g'] = np.exp(0.02+0.02*df)  
    df['v'] = np.where(df['g'] >= 1.0303, df['g']-1.0303, 2*(df['g']-1.0303))
    return df

rf_rate = 1.0303
df_monte_carlo = monte_carlo_weight(100000,1)
x_solution = np.zeros(100000)
R = ln_g(df_monte_carlo)
i = 0

for b1 in range(0,101,1):
    b0 = b1/10
    x_minus = 1
    x_plus = 1.1
    x_mid = 0.5*(x_minus+x_plus)
    x_mean = R['v'].mean()*R['g'].mean()*x_mid
    e_x = e_x = 0.99*b0*x_mean+0.99*x_mid-1
    while (np.abs(e_x) >= 10**(-4)):
        x_mid = 0.5*(x_minus+x_plus)
        x_mean = R['v'].mean()*R['g'].mean()*x_mid
        e_x = 0.99*b0*x_mean+0.99*x_mid-1
        if(e_x<0):
            x_minus = x_mid
        if(e_x>0):
            x_plus = x_mid
    else:
        x_answer = e_x
        i = i + 1

        
        
        
        
        
        
        
    
    
    

