# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:51:26 2022

@author: XuebinLi
"""

#hardcoded part is weight at line 56
#hardcoded part is lmda at 88
#hardcoded part is weight_Rtg2 at 159


import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MinuteLocator
import datetime
from datetime import date
from datetime import timedelta
import pandas as pd
from tabulate import tabulate
from numpy.linalg import inv
import math 
import plotly.express as px
np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)


# create dataframe
# df = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson5\\Industry_Portfolios.xlsx')
# df_market = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson5\\Market_Portfolio.xlsx')
df_monthly_returns = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\smu\\New folder\\Asset_Pricing_SMU\\Industry_Portfolios.xlsx') 
df_market = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\smu\\New folder\\Asset_Pricing_SMU\\Market_Portfolio.xlsx') 
df = df_monthly_returns.sub(df_market['Market'],axis=0)
d = []
d_without_date = []
for p in df:
    if "Date" not in p and "date" not in p:
        d.append((p, df[p].mean()))     
df_table_mean_std = pd.DataFrame(d, columns=('industry', 'expected_deviation'))



#covariance
df_cov = df.iloc[: , 1:]
df_cov = df_cov.cov()
#vector mean
vector_mean = df_table_mean_std[["expected_deviation"]].to_numpy() 
# transpose of returns
vector_mean_transpose = np.transpose(vector_mean)
#inverse covariance
df_cov_inverse = inv(df_cov)
#e
#weight = [1, 1, 1, 1, 1, 1, 1, 1, 1,1]
weight = np.ones(10)
#e transpose
weight_transpose = np.transpose(weight)


#alpha
alpha = np.matmul(vector_mean_transpose, df_cov_inverse)
alpha = np.matmul(alpha, weight)


#zetha
zelta = np.matmul(vector_mean_transpose, df_cov_inverse)
zelta = np.matmul(zelta, vector_mean)

#delta
delta = np.matmul(weight_transpose, df_cov_inverse)
delta = np.matmul(delta, weight)

#rmv mid line
rmv = alpha/delta

#riskfree_Rate
rf_rate = 0
    
    
#tangency portfolio
Rtg = (alpha * rf_rate - zelta)/(delta*rf_rate - alpha)
Rtg = Rtg[0][0]
Stg = -((zelta-2*alpha*rf_rate+delta*rf_rate*rf_rate)**0.5)/(delta*(rf_rate-rmv))


#sharpe ratio
def sharpe_ratio(Rtg,rf_rate,Stg):
    sharpe_ratio = (Rtg - rf_rate)/Stg
    return sharpe_ratio

#weight of optimal portfolio
def weight_portfolio():
    lmda = (Rtg - rf_rate)/(zelta - 2*alpha*rf_rate+delta*rf_rate*rf_rate)
    lmda = lmda[0][0]
    weight_Rtg1 = np.dot(lmda,df_cov_inverse)
    weight_Rtg1 = lmda * df_cov_inverse
    weight_Rtg2 = np.dot(rf_rate,weight)
    weight_Rtg2 = np.reshape(weight_Rtg2, (10, 1))
    weight_Rtg3 = np.subtract(vector_mean,weight_Rtg2)
    weight_final = np.dot(weight_Rtg1,weight_Rtg3)
    weight_pd = pd.DataFrame(data=weight_final, index = df.columns[1:11],columns=['Weight'])
    return weight_final, weight_pd


def print_all(sharpe_ratio,weight_final,vector_mean,df_cov,df_table_mean_std):    
    print("information ratio:", sharpe_ratio, "\n" )
    print("weight of optimal portfolio", "\n")
    print(weight_final[1], "\n")
    #question1: vector of mean and covariance
    print("covariance:", "\n")
    print(df_cov, "\n")
    #question2: table with mean and std
    print(tabulate(df_table_mean_std, headers = 'keys', tablefmt = 'psql'), "\n")
    
    
def plot_all():
    def my_range(start, end, step):
        while start <= end:
            yield start
            start += step

    #risk-free line(PAGE 25 lecture)
    yaxis2 = []
    xaxis2 = []
    for x2 in my_range(rf_rate, 0.11, 0.005):
        stdplot2 = ((x2 - rf_rate)**2)/(zelta - 2*alpha*rf_rate+zelta*rf_rate*rf_rate)
        stdplot2 = math.sqrt(stdplot2)
        xaxis2 += [stdplot2]
        yaxis2 += [x2]
    plt.plot(xaxis2,yaxis2,label="Tangent Line")
    plt.xlabel("Tracking Error")
    plt.ylabel("Expected Monthly Deviation ")
    plt.legend()   


    yaxis = []
    xaxis = []
    for x in my_range(0, 0.11, 0.005):
        stdplot = (1/delta) + (delta/(zelta*delta-(alpha*alpha))) * (x - (alpha/delta))**2
        stdplot = math.sqrt(stdplot)
        xaxis += [stdplot]
        yaxis += [x]
    plt.plot(xaxis,yaxis,label="minimum variance frontier")
    plt.xlabel("Tracking Error")
    plt.ylabel("Expected Monthly Deviation ")
    plt.xlim(0,0.25)
    plt.legend()           

#print_all(sharpe_ratio(Rtg,rf_rate,Stg),weight_portfolio(),vector_mean,df_cov,df_table_mean_std)
#plot_all()

#df.to_excel (r'C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson2\\std_mean.xlsx')
#df.to_excel (r'C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\std_mean.xlsx')



#monte carlo simulation

def monte_carlo():
    #get weight 100000 * 10 
    #sum = 1 and value>0
    matrix_weight = np.random.dirichlet(np.ones(10),size=(100000))
    # print(matrix_weight, "\n")
    df_matrix = pd.DataFrame(matrix_weight)
    df_matrix.columns = [df_monthly_returns.columns[1:11]]
    df_monthly_returns_np = df_monthly_returns.drop(['Date'], axis=1)
    df_monthly_returns2 = df_monthly_returns_np.mean()
    df_monthly_returns2_np = df_monthly_returns2.to_numpy()
    df_monthly_returns2_np = np.transpose(df_monthly_returns2_np)
    df_monthly_std2 = df_monthly_returns_np.std()
    df_monthly_std2_np = df_monthly_std2.to_numpy()
    df_monthly_std2_np = np.transpose(df_monthly_std2_np)    

    
    #weight x returns
    np_weight_return = matrix_weight*df_monthly_returns2_np
    df_np_weight_return = pd.DataFrame(data = np_weight_return)
    df_np_weight_return.to_csv('df_np_weight_return.csv')
    
    #std * returns
    np_std_return = matrix_weight*df_monthly_std2_np
    df_np_std_return = pd.DataFrame(data = np_std_return)
    df_np_std_return.to_csv('df_np_std_return.csv')
    #alpha
    alpha = np.matmul(vector_mean_transpose, df_cov_inverse)
    alpha = (alpha * matrix_weight)

    #zetha
    zelta = np.matmul(vector_mean_transpose, df_cov_inverse)
    zelta = (zelta * np_weight_return)
    
    #delta
    delta = np.matmul(weight_transpose, df_cov_inverse)
    delta = (delta * matrix_weight)    
    

    rmv = alpha/delta


    Rtg = (alpha * rf_rate - zelta)/(delta*rf_rate - alpha)
    Rtg = Rtg[0][0]
    Stg = -((zelta-2*alpha*rf_rate+delta*rf_rate*rf_rate)**0.5)/(delta*(rf_rate-rmv))
    
    sharpe_ratio_monte = sharpe_ratio(Rtg,rf_rate,Stg)
    print("sharpe ratio:","\n", sharpe_ratio_monte)
    df_sharpe_ratio_monte = pd.DataFrame(data = sharpe_ratio_monte)
    df_sharpe_ratio_monte.to_csv('tst.csv')

#    plt.axis([0, 0.5, 1, 1.5])    
    plt.scatter(np.array(np_std_return), np.array(np_weight_return), color='green')
    # Add labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # Display
    plt.show()
    
    
    # fig = px.scatter(x=np.array(np_std_return[0]), y=np.array(np_weight_return[0]))
    # fig.show()

    
    return df_matrix
     
monte_carlo()    
    















