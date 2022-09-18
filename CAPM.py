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
length = 10 #linear regression needs to be 10
vector_mean = AP_project1.vector_mean
x = vector_mean
rf_rate = AP_project1.rf_rate



def market_model(data_industry,data_mkt,rf):   
    #get excess returns
    y1 = data_industry - rf
    x1 = data_mkt - rf
    #fit excess returns into y axis for industrial and x axis for mkt
    rg_exrt = LinearRegression().fit(x1,y1)
    idt_alpha = rg_exrt.intercept_
    idt_beta = rg_exrt.coef_    
    rg_coefficient = pd.DataFrame(np.concatenate((idt_alpha.reshape(1,10),idt_beta.reshape(1,10))),
                                       index = ['intercept coefficient','slope coefficient'],
                                       columns = data_industry.columns)
    print(rg_coefficient)
    
    #regress mean    
    #mean market
    mean_mkt = np.array(data_mkt.mean())
    #mean of 10 industry
    mean_industry = np.array(data_industry.mean())
    #mean of market + industry
    mean_mkt_industry = np.concatenate((mean_mkt, mean_industry), axis=0)
    mean_mkt_industry = np.concatenate((mean_industry, mean_mkt), axis=0)
    #change to 11 by 1 array
    mean_mkt_industry = mean_mkt_industry.reshape(11,1)
    #hardcode beta array
    mkt_beta = [[1]]
    #beta of market + industry
    mkt_idt_beta = np.concatenate((mkt_beta, idt_beta), axis=0)
    mkt_idt_beta = np.concatenate((idt_beta, mkt_beta), axis=0)
    #regresssion of market+mean over market beta+industry beta
    rg_sml = LinearRegression().fit(mkt_idt_beta,mean_mkt_industry)    
    #sml intercept
    intercept_SML = rg_sml.intercept_
    #sml slope
    slope_SML = rg_sml.coef_    
    print("")
    print("sml intercept:", intercept_SML)
    print("sml slope:",slope_SML)


    #plot graph
    plt.scatter(mkt_beta,mean_mkt,c='r',label='Industry Portfolios')
    plt.scatter(idt_beta,mean_industry,c='g',label='Market Portfolio')

    yaxis = []
    xaxis = []    
    betas = [x/10 for x in range(21)]
    assetReturns = [slope_SML[0][0]*x + intercept_SML[0] for x in betas]
    plt.plot(betas,assetReturns)
    plt.xlabel("Beta")
    plt.ylabel("Industrial Mean Return")
    plt.title("Security Market Line")
        
market_model(df_industry,df_market,rf_rate)




