# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:27:24 2022

@author: XuebinLi
"""


"""
Session 4: Linear Factor Models

Performance Measurement

Risk_Factors.xlsx contains monthly observations of the risk-free rate and the three Fama–French risk factors, 
all expressed as a percentage. These observations cover the ten-year period from Jan 2004 through Dec 2013.

→ Using excess returns for the ten industry portfolios, calculate the following performance metrics:

Sharpe ratio
Sortino ratio (using risk-free rate as target)
Treynor ratio (using CAPM β)
Jensen's α
Three-factor α
The sample semi-variance can be estimated as:


 where Ri is return on industry portfolio and Rf is risk-free rate.

→ Create a table showing the performance metrics for the ten industry portfolios.

→ Plot your results as a bar chart for each performance metric.

→ Briefly explain the economic significance of each of the three performance ratios (but not α's).

Economic significance:

Sharpe ratio represents risk premium per unit of total risk:
Includes idiosyncratic risk, which penalises individual investments and non-diversified portfolios
Implicitly assumes normal returns, so cannot distinguish between return distributions 
with same variance but different skewness

Sortino ratio represents risk premium per unit of downside risk: can distinguish 
between asymmetric return distributions with same variance but different skewness

Treynor ratio represents risk premium per unit of market risk: ignores idiosyncratic risk
6as well as other types of systematic risk

"""

import math
from sklearn.linear_model import LinearRegression
from numpy.linalg import inv
from tabulate import tabulate
import pandas as pd
from datetime import timedelta
from datetime import date
import datetime
from matplotlib.dates import DateFormatter, MinuteLocator
import matplotlib.pyplot as plt
import numpy as np
import glob
import warnings
import CAPM_latest as CAPM
warnings.simplefilter("ignore", UserWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df_industries = pd.read_excel(
    'C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Industry_Portfolios.xlsx')
df_riskfactors = pd.read_excel(
    'C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Risk_Factors.xlsx')

#CAPM BETA from lesson 3
capm_beta = CAPM.market_model(CAPM.df_industry,CAPM.df_market,CAPM.rf_rate)

def std_mean_industries():
    # excess returns for 10 portfolios
    df_industries_excess_returns = df_industries.sub(
        df_riskfactors['Rf'].values, axis=0)
    # Remove date in columns
    if('Date' in df_industries_excess_returns.columns or 'date' in df_industries_excess_returns):
        df_industries_excess_returns = df_industries_excess_returns.drop('Date', axis=1)
    # mean of excess returns  
    excess_returns = df_industries_excess_returns
    mean_returns_industries_excess_returns = df_industries_excess_returns.mean()
    # convert mean returns of industries to 2d arrays
    mean_returns_industries_excess_returns = np.array(
        [mean_returns_industries_excess_returns.tolist()])
    # std of industries
    std_returns_industries_excess_returns = df_industries_excess_returns.std()
    # convert std to 2d arrays
    std_returns_industries_excess_returns = np.array(
        [std_returns_industries_excess_returns.tolist()])
    return mean_returns_industries_excess_returns, std_returns_industries_excess_returns, excess_returns


def excess_market_return(df_riskfactors):
    excess_market_returns = df_riskfactors['Rm-Rf'].mean()
    excess_market_returns_without_mean = df_riskfactors['Rm-Rf']
    excess_market_returns_without_rf_rate = df_riskfactors['Rm-Rf'] + df_riskfactors['Rf']
    #convert to 2d array
    excess_market_returns = np.array([[excess_market_returns]])
    SMB = df_riskfactors['SMB']
    HML = df_riskfactors['HML']
    return excess_market_returns, excess_market_returns_without_mean, SMB, HML, excess_market_returns_without_rf_rate


def sharpe_ratio(df_industries, df_riskfactors, mean_returns_industries_excess_returns, std_returns_industries_excess_returns):
    sharpe_ratio_industries = mean_returns_industries_excess_returns / std_returns_industries_excess_returns
    sharpe_ratio_industries = sharpe_ratio_industries.reshape(10,1)
    return sharpe_ratio_industries




def downside_risk(mean_returns_industries_excess_returns,excess_market_return):
    min_ri_minus_rt = mean_returns_industries_excess_returns - excess_market_return
    min_ri_minus_rt[min_ri_minus_rt>=0] = 0
    downside_risk = np.minimum(0,min_ri_minus_rt)**2
    return downside_risk


#wrong answers
def sortino_ratio(rp_minus_rf,rm_minus_rf):
    down_risk = np.mean(np.minimum(rp_minus_rf,0)**2)
    sortino = np.mean(rp_minus_rf)/np.sqrt(down_risk)
    sortino = np.array(sortino)
    return sortino


def jenson_alpha(mean_returns_industries_excess_returns,excess_market_return):
    mean_returns_industries_excess_returns = pd.DataFrame(mean_returns_industries_excess_returns)
    excess_market_return = pd.DataFrame(excess_market_return)
    reg = LinearRegression().fit(excess_market_return, mean_returns_industries_excess_returns)
    beta = reg.coef_
    alpha = reg.intercept_
    # print(alpha.ndim)
    alpha = alpha.reshape(10,1)
    return alpha

def three_factor_alpha(market_risk,SMB,HML,portfolio_excess_returns):
    
    market_risk = pd.DataFrame(market_risk)
    SMB = pd.DataFrame(SMB)
    HML = pd.DataFrame(HML)
    portfolio_excess_returns = pd.DataFrame(portfolio_excess_returns)
    df_all = pd.DataFrame()
    df_all['SMB'] = SMB
    df_all['HML'] = HML
    df_all['market_risk'] = market_risk
    reg = LinearRegression().fit(df_all,portfolio_excess_returns)
    alpha = reg.intercept_
    alpha = alpha.reshape(10,1)
    return alpha

def treynor_ratio(excess_market_return,excess_returns):
    all_port_excess_returns = pd.DataFrame(excess_returns)
    excess_market_return = pd.DataFrame(excess_market_return)
    reg = LinearRegression().fit(excess_market_return,all_port_excess_returns)
    beta = reg.coef_
    excess_return_mean = np.array(np.mean(excess_returns))
    print(excess_return_mean)
    print(beta)
    treynor_ratio = np.divide(excess_return_mean.reshape(10,1),beta.reshape(10,1))
    return treynor_ratio
    
        
def plot_chart(capm_industry):
    treynor_plot = treynor_ratio(excess_market_return(df_riskfactors)[1],std_mean_industries()[2])
    jenson_alpha_plot = jenson_alpha(std_mean_industries()[2],excess_market_return(df_riskfactors)[1])
    three_factor_plot = three_factor_alpha(excess_market_return(df_riskfactors)[1],excess_market_return(df_riskfactors)[2],excess_market_return(df_riskfactors)[3],std_mean_industries()[2])
    sharpe_ratio_plot = sharpe_ratio(df_industries, df_riskfactors,
                 std_mean_industries()[0], std_mean_industries()[1])
    sortino_ratio_plot = sortino_ratio(std_mean_industries()[2],excess_market_return(df_riskfactors)[1])
    plot_all_ratios = pd.DataFrame(np.concatenate((sharpe_ratio_plot.reshape(10,1),sortino_ratio_plot.reshape(10,1), \
                                                   treynor_plot.reshape(10,1), jenson_alpha_plot.reshape(10,1), three_factor_plot.reshape(10,1)),axis=1), \
                                               index = capm_industry.columns,
                                               columns = ['Sharpe Ratio','Sortino Ratio','Treynor Ratio','Jensen\'s Alpha','Three-Factor Alpha'] )               
    print(plot_all_ratios)
    plot_all_ratios.plot(y= ["Sharpe Ratio"], kind = "bar", color = 'red')
    plot_all_ratios.plot(y=["Sortino Ratio"], kind = "bar", color = 'green')
    plot_all_ratios.plot(y=['Treynor Ratio'], kind = 'bar', color = 'blue' )
    plot_all_ratios.plot(y=["Jensen's Alpha"], kind = "bar", color = 'brown')
    plot_all_ratios.plot(y=["Three-Factor Alpha"], kind = "bar", color = 'purple')
    



#call functions
plot_chart(CAPM.df_industry)
    

    
    
