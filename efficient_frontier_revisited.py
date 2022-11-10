# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 14:51:26 2022

@author: XuebinLi
"""

"""
Session 5: Efficient Frontier Revisited

Part 1: Minimum-Tracking-Error Frontier


"""

import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from numpy.linalg import inv
import math 

# create dataframe
df_monthly_returns = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Industry_Portfolios.xlsx')
df_market = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Market_Portfolio.xlsx')
#df_monthly_returns = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\smu\\New folder\\Asset_Pricing_SMU\\Industry_Portfolios.xlsx') 
#df_market = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\smu\\New folder\\Asset_Pricing_SMU\\Market_Portfolio.xlsx') 
df = df_monthly_returns.sub(df_market['Market'],axis=0)
d = []
d_without_date = []
for p in df:
    if "Date" not in p and "date" not in p:
        d.append((p, df[p].mean()))     
df_table_mean_std = pd.DataFrame(d, columns=('industry', 'expected_deviation'))


#riskfree_Rate
rf_rate = 0.0

"""
page 4
covariance matrix of returns against each of the industries
Let V be n × n covariance matrix of returns, which consists of
variances on diagonal and covariances on off-diagonal
inverse covariance
"""


#covariance
df_cov = df.iloc[: , 1:]
df_cov = df_cov.cov()
np_df_cov = df_cov.to_numpy()
df_cov_inverse = inv(df_cov)

"""
page 4
Let R = (R1, . . . , Rn)′ be n × 1 vector of expected returns
vector mean
R be nx1 vector of expected returns.
Total of 10 mean returns. Which means it is the returns of each column.
transpose of returns
"""

#vector mean
vector_mean = df_table_mean_std[["expected_deviation"]].to_numpy() 
# transpose of returns
vector_mean_transpose = np.transpose(vector_mean)
#inverse covariance


"""
page 5
weight to represent investor health allocated on each industries
1 here means they are equally weighted.
transpose the weight

"""

#e
#weight = [1, 1, 1, 1, 1, 1, 1, 1, 1,1]
weight = np.ones(10)
#e transpose
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

"""
page 13:
Rmv = α/δ is mean return for global minimum-variance portfolio
"""


#rmv mid line
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
    print("weight of tangency portfolio", "\n")
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

    # #risk-free line(PAGE 25 lecture)
    
    """
    page 25:
    x2 = rp
    (Rp − Rf )**2 / ζ − 2αRf + δR2
    """ 
    
    yaxis2 = []
    xaxis2 = []
    for x2 in my_range(rf_rate, 0.11, 0.005):
        stdplot2 = ((x2 - rf_rate)**2)/(zelta - 2*alpha*rf_rate+zelta*rf_rate*rf_rate)
        stdplot2 = math.sqrt(stdplot2)
        xaxis2 += [stdplot2]
        yaxis2 += [x2]
    plt.plot(xaxis2,yaxis2,label="Tangent Line")
    plt.xlabel("Monthly Tracking Error")
    plt.ylabel("Expected Monthly return Deviation ")
    plt.yticks(np.arange(0, max(yaxis2), 0.005))
    plt.legend()   

    """
    page 13 and page 11:
    Rmv = α / δ
    [1/δ] + [δ / ζδ − α2] * (Rp − Rmv )**2]

    """   

    yaxis = []
    xaxis = []
    for x in my_range(0, 0.105, 0.005):
        stdplot = (1/delta) + (delta/(zelta*delta-(alpha*alpha))) * (x - (alpha/delta))**2
        stdplot = math.sqrt(stdplot)
        xaxis += [stdplot]
        yaxis += [x]
    plt.plot(xaxis,yaxis,label="minimum tracking error frontier")
    plt.xlabel("Monthly Tracking Error")
    plt.ylabel("Expected Monthly return Deviation ")
    plt.yticks(np.arange(0, max(yaxis), 0.005))
    plt.xlim(0.0,0.25)
    plt.legend()           

#print_all(sharpe_ratio(Rtg,rf_rate,Stg),weight_portfolio(),vector_mean,df_cov,df_table_mean_std)
#plot_all()


"""
matrix_weight = 10 rows * 100,000 columns
then divide matrix_weight by the sum of all the weights.
This to ensure all weights add up to 1 for each row.

matrix_weight_dd = use 1 divide by matrix weight
Then divide by the sum of each row for all weights add up to 1.
"""

def monte_carlo_weight(sizee,industry):
    #get weight 100000 * 10 
    #sum = 1 and value>0
    matrix_weight = np.random.rand(industry,sizee)
    matrix_weight = matrix_weight/matrix_weight.sum(axis=0)
    matrix_weight_dd = np.divide(1,matrix_weight)
    matrix_weight_dd = matrix_weight_dd/matrix_weight_dd.sum(axis=0)
    
    return matrix_weight, matrix_weight_dd


"""
drop date from month returns
df_monthly_returns_np = portfolio returns minus market monthly returns
This is on each cell instead of the mean value
df_monthly_returns_numpy_mean = mean returns of all 10 industries.
returns_matrix = transpose matrix weight and them multiple by the monthly returns
the matrix weight is 10*100,000. After transpose it becomes 100,000 * 10.
100,000 * 10 vs 10 * 1. Thus returns_matrix is 100,000 * 1. 
df_cov_ri = covariance of monthly returns. This is all 100,000 values.

page22:
wprimexv = slice through the weight by rows and multiply by df_cov_ri(10x10)
wprimexv = 100,000 x 10
pt_var = pt_var(100000* 10) * matrix_weight.T slice to 10x10
"""


#monte carlo simulatio
def monte_carlo(sizee,industry,matrix_weight):    
    # print(matrix_weight, "\n")
    df_monthly_returns_np = df_monthly_returns.drop(['Date'], axis=1)   
    df_monthly_returns_numpy_mean = df_monthly_returns_np.mean().to_numpy()
    returns_matrix = np.matmul(matrix_weight.T,df_monthly_returns_numpy_mean)  
    df_returns_matrix = pd.DataFrame(data = returns_matrix)
    #df_returns_matrix_mean = df_returns_matrix.sum(axis=1)
    df_returns_matrix_mean = df_returns_matrix   
    df_returns_matrix_mean = df_returns_matrix_mean.to_numpy()
    df_cov_ri = df_monthly_returns_np.cov()
    df_cov_ri = df_cov_ri.to_numpy()
    wprimexv = np.dot(matrix_weight.T[None,:],df_cov_ri).reshape(sizee,industry)
    pt_var = (wprimexv * matrix_weight.T[None,:].reshape(sizee,industry)).sum(axis=1)  
    pt_std = np.sqrt(pt_var)
    plt.scatter(pt_std, returns_matrix)
    plt.title('Monte Carlo Simulation')
    plt.xlabel('standard deviation')
    plt.ylabel('Expected Returns')
    plt.show()
    
    return True

monte_carlo(100000,10,monte_carlo_weight(100000,10)[0])
monte_carlo(100000,10,monte_carlo_weight(100000,10)[1])
