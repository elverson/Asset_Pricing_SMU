# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:27:24 2022

@author: XuebinLi
"""


"""
Session 4: Linear Factor Models

Performance Measurement

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
#capm_beta = CAPM.market_model(CAPM.df_industry,CAPM.df_market,CAPM.rf_rate)

def std_mean_industries():
    # excess returns for 10 portfolios
    df_industries_excess_returns = df_industries.sub(
        df_riskfactors['Rf'].values, axis=0)
    
    # Remove date in columns
    if('Date' in df_industries_excess_returns.columns or 'date' in df_industries_excess_returns):
        df_industries_excess_returns = df_industries_excess_returns.drop('Date', axis=1)
    
    """    
    excess_returns : set df_industries_excess_returns to excess_returns. 
    This is individual cell industries return minus rf
    
    mean_returns_industries_excess_returns : mean of excess returns. 
    individually subtract rf values from each cell of industries
    and then take mean of each column
    
    mean_returns_industries_excess_returns : convert mean returns of industries to 2d arrays
    """
    excess_returns = df_industries_excess_returns
    mean_returns_industries_excess_returns = df_industries_excess_returns.mean()
    mean_returns_industries_excess_returns = np.array(
        [mean_returns_industries_excess_returns.tolist()])
    
    """
    std_returns_industries_excess_returns : mean excess std of industries. 
    Return minus rf and take std()
    
    std_returns_industries_excess_returns: convert std to 2d arrays    
    """
    std_returns_industries_excess_returns = df_industries_excess_returns.std()
    std_returns_industries_excess_returns = np.array(
        [std_returns_industries_excess_returns.tolist()])
    
    """
    [0] = mean returns of industries return minus rf. 10 values
    [1] = std of industries minus rf. 10 values
    [2] = all values in industries data frame of returns minus rf
    """
    return mean_returns_industries_excess_returns, std_returns_industries_excess_returns, excess_returns


def excess_market_return(df_riskfactors):
    
    """
    excess_market_returns = mean of rm-rf of riskfactor excel sheet. This is market premium
    excess_market_returns_without_mean = all rm-rf values of the column without taking mean. 
    number of values is number of rows.
    excess_market_returns_without_rf_rate = this is just the market return without minus riskfree rate
    excess_market_returns = convert mean of rm-rf of riskfactor excel sheet to 2d array
    SMB = set df_riskfactors['SMB'] as all SMB values in column
    HML = set df_riskfactors['HML'] as all HML values in column
    """
    
    excess_market_returns = df_riskfactors['Rm-Rf'].mean()
    excess_market_returns_without_mean = df_riskfactors['Rm-Rf']
    excess_market_returns_without_rf_rate = df_riskfactors['Rm-Rf'] + df_riskfactors['Rf']
    #convert to 2d array
    excess_market_returns = np.array([[excess_market_returns]])
    SMB = df_riskfactors['SMB']
    HML = df_riskfactors['HML']
    "“UMD under page 7 of lecture slides”"
    
    """
    [0] = mean of rm-rf of riskfactor excel sheet. This is market premium
    [1] = all rm-rf values of the column without taking mean in column 
    [2] = all SMB values in column
    [3] = all HML values in column
    [4] = this is just the market return without minus riskfree rate and did not take mean. 
    All values in column
    
    """
    return excess_market_returns, excess_market_returns_without_mean, SMB, HML, excess_market_returns_without_rf_rate


    """
    page 10:
    Si = E(˜Ri −Rf) / sqrt(Var(˜Ri − Rf))    
        
    sharpe
    [0] = excel of industries portfolio
    [1] = excel of riskfactors
    [2] = mean of excess returns. individually subtract rf values from each cell of industries
    [3] = mean excess std of industries. Return minus rf and take std()
    
    sharpe_ratio_industries = mean of 10 industries over std of 10 industries. rf rate is minus of already.

    """

def sharpe_ratio(df_industries, df_riskfactors, mean_returns_industries_excess_returns, std_returns_industries_excess_returns):
    sharpe_ratio_industries = mean_returns_industries_excess_returns / std_returns_industries_excess_returns
    sharpe_ratio_industries = sharpe_ratio_industries.reshape(10,1)
    return sharpe_ratio_industries


    """
    Page 14 and page 15:
    Sti = E(˜Ri − ˜Rt) / sqrt(SV (˜Ri;˜Rt))
    
    sortino
    [0] = This is individual cell industries return minus rf all values
    [1] = all rm-rf values of the column without taking mean. 
    
    downside risk = individual cell industries return minus rf. if negative take value 0. 
    if not use the current value. Square it and lastly take mean . total 10 industry values.
    
    sortino = all industries return minus rf all values divide by square root of downside risk and
    take mean. Total 10 values.
    
    """

def sortino_ratio(rp_minus_rf,rm_minus_rf):
    down_risk = np.mean(np.minimum(rp_minus_rf,0)**2)
    sortino = np.mean(rp_minus_rf)/np.sqrt(down_risk)
    sortino = np.array(sortino)
    return sortino


    """
    page 12:
    αi = E(˜Ri − Rf)− βi * E(˜Rm − Rf)
    
    (Ri - rf) is from all cell values in industires excel file
    (rm -rf) is all values in the column from riskfactor excel
    
    Regress all values of (rI - rf) over all values of (rm - rf)
    and get the alpha which is the intercept.    
    """

def jenson_alpha(mean_returns_industries_excess_returns,excess_market_return):
    mean_returns_industries_excess_returns = pd.DataFrame(mean_returns_industries_excess_returns)
    excess_market_return = pd.DataFrame(excess_market_return)
    reg = LinearRegression().fit(excess_market_return, mean_returns_industries_excess_returns)
    #beta = reg.coef_
    alpha = reg.intercept_
    # print(alpha.ndim)
    alpha = alpha.reshape(10,1)
    return alpha


    """
    page 3:
    Eugene Fama and Kenneth French use three-factor model
    with risk factors for market risk, size risk, and value risk:  
    Size risk = SMB
    value risk = HMB
    market risk = market_risk
    Regress all this factors against the portfolio premium which is all values
    of (ri - rf)    
    lastly get alpha value
    """

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

    
    """
    page 11:
    Ti =E(˜Ri − Rf) / βi
    excess_market_return = all values of rm-rf
    excess_returns = all values of ri - rf
    regress all values of rm-rf over all values of ri - rf to get beta of industries returns
    Treynor ratio = all values of ri - rf divide by its market beta
    """


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
    

    
    
