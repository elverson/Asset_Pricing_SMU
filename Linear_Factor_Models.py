# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:27:24 2022

@author: XuebinLi
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
    'C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson4\\Industry_Portfolios.xlsx')
df_riskfactors = pd.read_excel(
    'C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson4\\Risk_Factors.xlsx')

#CAPM BETA from lesson 3
capm_beta = CAPM.capm_beta(CAPM.df_industry,CAPM.df_market,CAPM.rf_rate)

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
    print(rm_minus_rf)
    print(rp_minus_rf)
    down_risk = np.mean(np.minimum(rp_minus_rf,0)**2)
    sortino = np.mean(rp_minus_rf)/np.sqrt(down_risk)
    sortino = np.array(sortino)
    return sortino


def jenson_alpha(mean_returns_industries_excess_returns,excess_market_return):
    mean_returns_industries_excess_returns = pd.DataFrame(mean_returns_industries_excess_returns)
    excess_market_return = pd.DataFrame(excess_market_return)
    reg = LinearRegression().fit(excess_market_return, mean_returns_industries_excess_returns)
    #beta = reg.coef_
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

def treynor_ratio(CAPM_beta, mean_excess_return):
    mean_excess_return = mean_excess_return.reshape(10,1)
    treynor_ratio = np.divide(mean_excess_return,CAPM_beta)
    return treynor_ratio
    


def plot_chart(capm_industry):
    treynor_plot = treynor_ratio(capm_beta,std_mean_industries()[0])
    jenson_alpha_plot = jenson_alpha(std_mean_industries()[2],excess_market_return(df_riskfactors)[1])
    three_factor_plot = three_factor_alpha(excess_market_return(df_riskfactors)[1],excess_market_return(df_riskfactors)[2],excess_market_return(df_riskfactors)[3],std_mean_industries()[2])
    sharpe_ratio_plot = sharpe_ratio(df_industries, df_riskfactors,
                 std_mean_industries()[0], std_mean_industries()[1])
    sortino_ratio_plot = sortino_ratio(std_mean_industries()[2],excess_market_return(df_riskfactors)[1])
    plot_all_ratios = pd.DataFrame(np.concatenate((treynor_plot.reshape(10,1),jenson_alpha_plot.reshape(10,1), \
                                                   three_factor_plot.reshape(10,1), sharpe_ratio_plot.reshape(10,1), sortino_ratio_plot.reshape(10,1)),axis=1), \
                                               index = capm_industry.columns,
                                               columns = ['Treynor Ratio','Jenson Alpha','Three-factor Alpha','Sharpe Ratio','Sortino Ratio'] )
    
    print(plot_all_ratios)


plot_chart(CAPM.df_industry)



#treynor_ratio(capm_beta,std_mean_industries()[0])
#jenson_alpha(std_mean_industries()[2],excess_market_return(df_riskfactors)[1])
#three_factor_alpha(excess_market_return(df_riskfactors)[1],excess_market_return(df_riskfactors)[2],excess_market_return(df_riskfactors)[3],std_mean_industries()[2])
#sharpe_ratio(df_industries, df_riskfactors,
             #std_mean_industries()[0], std_mean_industries()[1])
#sortino_ratio(std_mean_industries()[2],excess_market_return(df_riskfactors)[1])



    

    
    
