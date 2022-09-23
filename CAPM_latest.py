# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:27:24 2022

@author: XuebinLi
"""
import warnings
warnings.simplefilter("ignore", UserWarning)
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
import asset_pricing_efficient_frontier_enhanced as AP_project1
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#set up dataframes
df_industry = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson3\\Industry_Portfolios.xlsx',index_col=0,header=0)
df_market = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\lesson3\\Market_Portfolio.xlsx',index_col=0,header=0)
#df = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\Project2\\Industry_Portfolios.xlsx')
#df2 = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\Project2\\Market_Portfolio.xlsx')

#Reload modules: asset_pricing_efficient_frontier_enhanced
AP_project1

#declaration
#risk-free rate
rf_rate = AP_project1.rf_rate



def market_model(data_industry,data_mkt,rf):   
    #first part is capm model    
    #portfolio risk premium
    RP_minus_RF = df_industry- rf_rate
    #market risk premium
    RM_minus_RF = df_market - rf_rate
    #regress portfolio premium over market premium
    reg = LinearRegression().fit(RM_minus_RF, RP_minus_RF)
    #industry beta which is the slope
    industry_beta = reg.coef_
    #intercept which is the alpha
    industry_alpha = reg.intercept_
    #create table of alpha and beta of 10 industries.
    market_coefficient = pd.DataFrame(np.concatenate((industry_alpha.reshape(1,10),industry_beta.reshape(1,10))),
                                       index = ['intercept coefficient','slope coefficient'],
                                       columns = data_industry.columns)
    print(market_coefficient)

    #this part is security market line
    #average return of each of the 10 industrial portfolios
    mean_industry_returns = df_industry.mean()
    #average return of the market returns
    mean_market_returns = df_market.mean()
    #total average returns of portfolio and market
    mean_industry_plus_mkt_returns = np.concatenate((mean_industry_returns, mean_market_returns),axis=0)
    #market beta= 1
    market_beta = [[1]]
    #10 industry beta + 1 market beta
    mean_industry_plus_market_beta = np.concatenate((industry_beta, market_beta),axis=0)    
    #regress average portfolio + market returns over average portfolio + market betas
    reg_sml = LinearRegression().fit(mean_industry_plus_market_beta, mean_industry_plus_mkt_returns)
    #beta of sml
    reg_sml_coefficient_slope = reg_sml.coef_
    #alpha of sml
    reg_sml_intercept = reg_sml.intercept_    
    print("sml intecept:",reg_sml_intercept)
    print("sml slope:",reg_sml_coefficient_slope[0])
    
    #plot sml
    #for loop over 0-2 x axis with constant of m and c    
    def my_range(start, end, step):
        while start <= end:
            yield start
            start += step
    yaxis = []
    xaxis = []
    for x in my_range(0, 2.1, 0.1):
        stdplot = reg_sml_coefficient_slope*x + reg_sml_intercept
        xaxis += [x]
        yaxis += [stdplot]
    plt.plot(xaxis,yaxis) 
    plt.xlabel("Beta")
    plt.ylabel("Returns")
    plt.title("Security Market Line")
    plt.scatter(market_beta,mean_market_returns,c='r',label='Industry Portfolios')
    plt.scatter(industry_beta,mean_industry_returns,c='g',label='Market Portfolio')

    
    
market_model(df_industry,df_market,rf_rate)




