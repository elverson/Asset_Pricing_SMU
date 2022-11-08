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
    x_bi = {"x-":1, "x+":1.1}
    ex = {"x-":np.nan,"x+":np.nan,"x":1}
    b0 = b1/10
    while abs(ex["x"]) >= 10**(-4):
        x = (x_bi["x-"]+x_bi["x+"])/2
        x_bi["x"] = x
        
        
        
        
        
        
        
        
    
    
    

