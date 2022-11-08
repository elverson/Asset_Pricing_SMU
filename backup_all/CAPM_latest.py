# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:27:24 2022

@author: XuebinLi
"""

"""
Session 3: Capital Asset Pricing Model (CAPM)

Market_Portfolio.xlsx contains monthly nominal (net) returns for the market portfolio, 
expressed as a percentage. These returns cover the ten-year period from Jan 2004 through Dec 2013. 
Assume that the (net) risk-free rate is 0.13% per month.

Market Model

Estimate the intercept coefficient (α) and slope coefficient (β) for each of the 
ten industry portfolio using the market model: regress the monthly excess returns for each 
industry portfolio on the monthly excess returns for the market portfolio.

→ Create a table showing the intercept and slope coefficients for the ten industry portfolios.

→ Briefly explain the economic significance of the intercept and slope coefficients.

Security Market Line (SML)

Calculate the mean monthly return for each of the ten industry portfolios, as well as the market portfolio.

Regress the mean monthly returns of the ten industry portfolios and the 
market portfolio on the corresponding β's. This will give you the 
intercept and slope coefficients for the SML. 
(Warning: the results may be very different from what you would expect!)

→ Use the estimated intercept and slope coefficients for the SML to plot the SML in the range of β from zero to two on the horizontal axis. Also plot the positions of the ten industry portfolios and the market portfolio. (You are NOT required to label the individual portfolios.)

→ Briefly explain the economic significance of the SML.



Economic significance:

Intercept coefficient from market model regression represents CAPM pricing error.

Slope coefficient from market model regression represents degree of exposure to market risk.

SML shows that all appropriately-priced assets have same Treynor ratio 
⇒ assets that lie above SML are underpriced, while assets that lie below SML are overpriced.
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
rf_rate = 0.13



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
    industry_market_coefficient = pd.DataFrame(np.concatenate((industry_alpha.reshape(10,1),industry_beta.reshape(10,1)),axis=1),
                                               index = data_industry.columns,
                                               columns = ['intercept coefficient','slope coefficient'] )
    

    print(tabulate(industry_market_coefficient, headers = 'keys', tablefmt = 'psql'))

    #this part is security market line
    #average return of each of the 10 industrial portfolios
    mean_industry_returns = df_industry.mean()
    #average return of the market returns
    mean_market_returns = df_market.mean()
    #total average returns of portfolio and market
    mean_industry_plus_mkt_returns = np.concatenate((mean_industry_returns, mean_market_returns),axis=0)
    #print mean returns of industry and market
    index_mkt_industry = data_industry.columns.insert(10, 'Market')
    mean_industry_plus_mkt_returns_print = pd.DataFrame(mean_industry_plus_mkt_returns.reshape(11,1), index = index_mkt_industry, 
                                                        columns=['Mean Returns'])
    print(tabulate(mean_industry_plus_mkt_returns_print,headers='keys',tablefmt='psql'))
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




