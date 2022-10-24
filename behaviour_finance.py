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
    df['r'] = np.exp(0.02+0.02*df)  
    df['v'] = np.where(df['r'] >= 1.0303, df['r']-1.0303, 2*(df['r']-1.0303))
    return df

df_monte_carlo = monte_carlo_weight(100000,1)
R = ln_g(df_monte_carlo)

