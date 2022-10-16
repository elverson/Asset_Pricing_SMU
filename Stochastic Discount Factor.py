# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 09:21:01 2022

@author: XuebinLi
"""
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

def monte_carlo_weight(column,rows):
    #get weight 100000 * 10 
    #sum = 1 and value>0
    v = np.random.uniform(0,1,size=(column, rows))    
    e = np.random.standard_normal(size=(column, rows))
    e = pd.DataFrame(e)
    #set to 0 if less than 0.17 
    #set to ln0.65 if more than 0.983
    v = pd.DataFrame(v)  
    v[v < 0.017] = np.log(0.65)    
    v[v >= 0.017] = 0
    return v, e


def plot_all():
    print(tabulate(smallest_1, headers = 'keys', tablefmt = 'psql'), "\n")
    print(tabulate(df_mean, headers = 'keys', tablefmt = 'psql'), "\n")
    print(tabulate(df_std, headers = 'keys', tablefmt = 'psql'), "\n")
    print(tabulate(m_std_divide_mean_df_print, headers = 'keys', tablefmt = 'psql'), "\n")   
    plt.plot(m_std_divide_mean_df)
    plt.xlabel('γ')
    plt.ylabel('σM/μM')
    plt.title('σM/μM vs γ')
    plt.xlim(1,4)
    plt.show()


e =  monte_carlo_weight(100000,1)[1]
v = monte_carlo_weight(100000,1)[0]  
#ln_g = 0.02+0.02e+v
#df_ln_g = 0.02 + e*0.02 + v
df_ln_g = np.exp(0.02 + 0.02*e + v)
ln_g = df_ln_g.to_numpy()

y = np.arange(1,4.1,0.1)
y_index = y
y_df = pd.DataFrame(data = y,index=y)
y = y.reshape(1,31)

#M = 0.99*df_ln_g**(-Y)
g_from_ln_g = df_ln_g
m = np.power(ln_g, -y) * 0.99
pd_m = pd.DataFrame(data = m)
mean_m = pd_m.mean()
std_m = pd_m.std()
m_std_divide_mean = (np.divide(std_m,mean_m))
m_std_divide_mean_df_print = pd.DataFrame(data = m_std_divide_mean, columns=['σM/μM'])
m_std_divide_mean_df_print.index.name= 'γ'
m_std_divide_mean_df = m_std_divide_mean_df_print.set_index(y_df.iloc[:,0])
smallest = m_std_divide_mean_df[m_std_divide_mean_df > 0.4]
smallest_1 = smallest.dropna().iloc[:1]
smallest_1.columns=['σM/μM > 0.4']
df_std = pd.DataFrame(data = std_m, columns=['standard deviation'])
df_mean = pd.DataFrame(data = mean_m, columns=['mean'])
df_mean = df_mean.set_index(y_index)
df_std = df_std.set_index(y_index)
smallest = smallest.set_index(y_index)
m_std_divide_mean_df_print = m_std_divide_mean_df_print.set_index(y_index)
df_mean.index.name='γ'
df_std.index.name='γ'
smallest.index.name='γ'
smallest_1.index.name='γ'
plot_all()





    


