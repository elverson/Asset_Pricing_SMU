
"""
Created on Thu Sep  1 14:51:26 2022

@author: XuebinLi
"""

"""

Session 2: Efficient Frontier
The purpose of this exercise is to 


"""

#hardcoded part is weight at line 95
#hardcoded part is lmda at 134
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
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#riskfree_Rate
rf_rate = 0.13

# create dataframe
df = pd.DataFrame()
df = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Industry_Portfolios.xlsx')
#df = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\project2\\Industry_Portfolios.xlsx') 
d = []

#exclude date from industry portfolio
for p in df:
    if "Date" not in p and "date" not in p:
        d.append((p, df[p].mean(), df[p].std()))
df_table_mean_std = pd.DataFrame(d, columns=('industry', 'mean_return', 'standard_deviation'))

"""
page 4
covariance matrix of returns against each of the industries
Let V be n × n covariance matrix of returns, which consists of
variances on diagonal and covariances on off-diagonal
inverse covariance
"""
df_cov = df.iloc[: , 1:]
df_cov = df_cov.cov()
df_cov_inverse = inv(df_cov)

"""
page 4
Let R = (R1, . . . , Rn)′ be n × 1 vector of expected returns
vector mean
R be nx1 vector of expected returns.
Total of 10 mean returns. Which means it is the returns of each column.
transpose of returns
"""
vector_mean = df_table_mean_std[["mean_return"]].to_numpy() 
vector_mean_transpose = np.transpose(vector_mean)

"""
page 5
weight to represent investor health allocated on each industries
1 here means they are equally weighted.
transpose the weight

"""
weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
weight_transpose = np.transpose(weight)


"""
page 11:
formula for alpha
α = R′V−1e; 
formula for zelta
ζ = R′V−1R; 
formula for delta
δ = e′V−1e
"""
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
"""
page 13:
Rmv = α/δ is mean return for global minimum-variance portfolio
"""
rmv = alpha/delta
  
"""
page 28:
Rtg − Rf = (αRf − ζ) /  (δRf − α)  - Rf
Rtg = Rtg[0][0] to get exact value instead of an array.

page 29:
σtg = − [(ζ − 2αRf + δR2)^0.5 / (δ (Rf − Rmv ))]

"""
#tangency portfolio
Rtg = (alpha * rf_rate - zelta)/(delta*rf_rate - alpha)
Rtg = Rtg[0][0]
Stg = -((zelta-2*alpha*rf_rate+delta*rf_rate*rf_rate)**0.5)/(delta*(rf_rate-rmv))

"""
page 30:
(Rtg − Rf) / σtg

"""
#sharpe ratio
def sharpe_ratio(Rtg,rf_rate,Stg):
    sharpe_ratio = (Rtg - rf_rate)/Stg
    return sharpe_ratio


"""
page 24: 
w∗ = λV−1 (R − Rf e)
λ = Rp − Rf / ζ − 2αRf + δ * Rf * 2

w∗ = λV−1 (R − Rf e)
"""

#weight of optimal portfolio
def weight_portfolio():
    lmda = (Rtg - rf_rate)/(zelta - 2*alpha*rf_rate+delta*rf_rate*rf_rate)
    lmda = lmda[0][0]
    lmda = [lmda,lmda,lmda,lmda,lmda,lmda,lmda,lmda,lmda,lmda]
    weight_Rtg1 = np.dot(lmda,df_cov_inverse)
    weight_Rtg1 = lmda * df_cov_inverse
    weight_Rtg2 = np.dot(rf_rate,weight)
    weight_Rtg2 = np.reshape(weight_Rtg2, (10, 1))
    weight_Rtg3 = np.subtract(vector_mean,weight_Rtg2)
    weight_final = np.dot(weight_Rtg1,weight_Rtg3)
    return weight_final


def print_all(sharpe_ratio,weight_final,vector_mean,df_cov,df_table_mean_std):    
    print("sharpe_ratio:", sharpe_ratio )
    print("weight of optimal portfolio")
    print(weight_final)
    #question1: vector of mean and covariance
    print("vector_mean:")
    print(vector_mean)
    print("covariance:")
    print(df_cov)
    #question2: table with mean and std
    print(tabulate(df_table_mean_std, headers = 'keys', tablefmt = 'psql'))
    #df_table_mean_std.plot(x='standard_deviation', y='mean_return', kind='scatter');
    
 
"""
page 13 and page 11:
Rmv = α / δ
[1/δ] + [δ / ζδ − α2] * (Rp − Rmv )**2]
"""    

def plot_all():
    def my_range(start, end, step):
        while start <= end:
            yield start
            start += step
    #minimum variance frontier(page 11 and 13 lecture)
    yaxis = []
    xaxis = []
    for x in my_range(0, 2.1, 0.1):
        stdplot = (1/delta) + (delta/(zelta*delta-(alpha*alpha))) * (x - (alpha/delta))**2
        stdplot = math.sqrt(stdplot)
        xaxis += [stdplot]
        yaxis += [x]
    plt.plot(xaxis,yaxis,label="Without Riskless Asset")
    plt.xlabel("Standard deviation")
    plt.ylabel("Returns in percentage")
    plt.title("minimum variance frontier")
    plt.legend()

    """
    page 25:
    x2 = rp
    (Rp − Rf )**2 / ζ − 2αRf + δR2
    """    
    yaxis2 = []
    xaxis2 = []
    for x2 in my_range(rf_rate, 2.1, 0.1):
        stdplot2 = ((x2 - rf_rate)**2)/(zelta - 2*alpha*rf_rate+zelta*rf_rate*rf_rate)
        stdplot2 = math.sqrt(stdplot2)
        xaxis2 += [stdplot2]
        yaxis2 += [x2]
    plt.plot(xaxis2,yaxis2,label="With Riskless Asset")
    plt.xlabel("Standard deviation")
    plt.ylabel("Returns in percentage")
    plt.xlim(0,6)
    plt.legend()   

    # #mid line
    # yaxis3 = []
    # xaxis3 = []
    # for x3 in my_range(rf_rate, 10, 0.1):
    #     stdplot3 = alpha/delta
    #     xaxis3 += [stdplot3]
    #     yaxis3 += [x3]
    # plt.plot(yaxis3,xaxis3,label='Rmv')
    # plt.xlabel("Standard deviation")
    # plt.ylabel("Returns in percentage")
    # plt.legend()


    # #efficient frontier 
    # yaxis4 = []
    # xaxis4 = []
    # for x4 in my_range(1.004, 2.1, 0.1):
    #     stdplot4 = (1/delta) + (delta/(zelta*delta-(alpha*alpha))) * (x4 - (alpha/delta))**2
    #     stdplot4 = math.sqrt(stdplot4)
    #     xaxis4 += [stdplot4]
    #     yaxis4 += [x4]
    # plt.plot(xaxis4,yaxis4 ,label="Without Riskless Asset")
    # plt.xlabel("Standard deviation")
    # plt.ylabel("Returns in percentage")
    # plt.title("efficient frontier")
    # plt.legend()      

#print_all(sharpe_ratio(Rtg,rf_rate,Stg),weight_portfolio(),vector_mean,df_cov,df_table_mean_std)
#plot_all()

#df.to_excel (r'C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson2\\std_mean.xlsx')
#df.to_excel (r'C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\std_mean.xlsx')

