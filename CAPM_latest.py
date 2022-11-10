# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:27:24 2022

@author: XuebinLi
"""

"""
Session 3: Capital Asset Pricing Model (CAPM)

market model:
Regress to find out the relationship of beta and alpha of 
10 industries return premium over market premium.

SML model:
Use the beta get from earlier to regress against the market beta.
The market beta is just 1. This gives us a new beta and alpha.
Plot the SML with respect to the value of market beta.
Y axis value = beta of industries * market beta(from 0 -2) 
+ alpha which is the intercet
    
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
df_industry = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Industry_Portfolios.xlsx',index_col=0,header=0)
df_market = pd.read_excel('C:\\Users\\lixue\\OneDrive\\Desktop\\smu\\MQF\\Asset Pricing\\github\\Asset_Pricing_SMU\\Market_Portfolio.xlsx',index_col=0,header=0)
#df = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\Project2\\Industry_Portfolios.xlsx')
#df2 = pd.read_excel('C:\\Users\\XuebinLi\\OneDrive - Linden Shore LLC\\Desktop\\python\\asset_pricing_project\\Project2\\Market_Portfolio.xlsx')

#Reload modules: asset_pricing_efficient_frontier_enhanced
#AP_project1

#declaration
#risk-free rate
#rf_rate = AP_project1.rf_rate

#set default risk-free rate
rf_rate = 0.13



def market_model(data_industry,data_mkt,rf):   
        
    """
    page 7:
    Rp − Rf = w′ (R − Rf*e)
    = w′⃗β (Rm − Rf ) = βp (Rm − Rf )
    βp = w′⃗
    
    """
    
    #first part is capm model    
    #getting portfolio risk premium   
    
    RP_minus_RF = df_industry- rf_rate
    
    #market risk premium
    RM_minus_RF = df_market - rf_rate
    
    """
    page 18: market model
    ˜Ri − Rf = αi + βi(˜Rm − Rf)+ ˜ϵi    
    
    Market model is one-factor linear regression model with
    (excess) asset return as dependent variable and (excess)
    market return as explanatory variable
    
    """
    #regress portfolio premium over market premium
    #market premium on x-axis and industry premium on y-axis
    #it is the measuring how portfolio premium is related to market premium
    reg = LinearRegression().fit(RM_minus_RF, RP_minus_RF)
    
    #industry beta which is the slope
    #As the beta of market go up, we see if the beta of industries go up or go down.
    industry_beta = reg.coef_
    
    #intercept which is the alpha
    #This is the extra y value from 0. If intercept is 0.4 then the alpha is 0.4.
    #alpha is the intercept of y axis
    industry_alpha = reg.intercept_
    
    #create table of alpha and beta of 10 industries. and print table
    industry_market_coefficient = pd.DataFrame(np.concatenate((industry_alpha.reshape(10,1),industry_beta.reshape(10,1)),axis=1),
                                               index = data_industry.columns,
                                               columns = ['intercept coefficient','slope coefficient'] )
    

    print(tabulate(industry_market_coefficient, headers = 'keys', tablefmt = 'psql'))

    #this part is security market line
    
    #average return of each of the 10 industrial portfolios
    mean_industry_returns = df_industry.mean()
    
    #average return of the market returns
    mean_market_returns = df_market.mean()
    
    #average returns of portfolio plus market
    mean_industry_plus_mkt_returns = np.concatenate((mean_industry_returns, mean_market_returns),axis=0)
    
    #add one more market index to the industry index
    #set as dataframe and print
    index_mkt_industry = data_industry.columns.insert(10, 'Market')
    mean_industry_plus_mkt_returns_print = pd.DataFrame(mean_industry_plus_mkt_returns.reshape(11,1), index = index_mkt_industry, 
                                                        columns=['Mean Returns'])
    print(tabulate(mean_industry_plus_mkt_returns_print,headers='keys',tablefmt='psql'))
    
    #set market beta= 1
    market_beta = [[1]]
    
    #set 10 industry beta + 1 market beta
    mean_industry_plus_market_beta = np.concatenate((industry_beta, market_beta),axis=0)  
    
    #regress average portfolio + market returns over average portfolio + market betas
    #it is measuring how industry return and market return is correlated with industry and market beta.
    reg_sml = LinearRegression().fit(mean_industry_plus_market_beta, mean_industry_plus_mkt_returns)
    
    #beta of sml which is the returns of industries and market over beta of industries and market
    #beta is the slope
    reg_sml_coefficient_slope = reg_sml.coef_
    
    #alpha of sml which is the extra returns of market + industries over beta.
    #alpha is the intercept of y axis.
    reg_sml_intercept = reg_sml.intercept_    
    print("sml intecept coefficient:",reg_sml_intercept)
    print("sml slope coefficient:",reg_sml_coefficient_slope[0])
    
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
    plt.ylabel("Expected Returns")
    plt.title("Security Market Line")
    plt.scatter(market_beta,mean_market_returns,c='r',label='Industry Portfolios')
    plt.scatter(industry_beta,mean_industry_returns,c='g',label='Market Portfolio')
    plt.xlim(0,2)

    
    
market_model(df_industry,df_market,rf_rate)




