# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:54:05 2022

@author: lixue
"""

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


risk_free_rate = 1.1
initial_share_price = 45
option_to_buy_A_B = 100

#final share price after one year
#header_state = good, normal, bad
#column = phys_prob, stock A, stock B

final_share_price = [[0.3, 0.5, 0.2], 
    [75, 55, 40],
    [60, 50, 40]]
final_share_price = np.array(final_share_price)

#vector of state prices
oneXthree_matrix = [[1/1.1,45,45]]
oneXthree_matrix = np.array(oneXthree_matrix)

threeXthree_matrix = [[1, 75, 60], 
    [1, 55, 50],
    [1, 20, 40]]

threeXthree_matrix = np.array(threeXthree_matrix)
threeXthree_matrix_inverse = np.linalg.inv(threeXthree_matrix)
vector_of_state_prices = np.matmul(oneXthree_matrix,threeXthree_matrix_inverse)
vector_of_risk_neutral_probabilities = vector_of_state_prices*risk_free_rate
#print(vector_of_state_prices)
#print(vector_of_risk_neutral_probabilities)

pay_off_vector = [max(threeXthree_matrix[0,1]+threeXthree_matrix[0,2]-100,0), 
    max(threeXthree_matrix[1,1]+threeXthree_matrix[1,2]-100,0),
    max(threeXthree_matrix[2,1]+threeXthree_matrix[2,2]-100,0)]
pay_off_vector = np.array(pay_off_vector)
call_price = pay_off_vector[0] * vector_of_state_prices[0,0] + pay_off_vector[1] * \
    vector_of_state_prices[0,1] + pay_off_vector[2] * vector_of_state_prices[0,2]


print("vector_of_state_prices: ", vector_of_state_prices, "\n")
print("vector_of_risk_neutral_probabilities: ", vector_of_risk_neutral_probabilities, "\n")
print("pay_off_vector:", pay_off_vector, "\n")
print("call_price:", call_price, "\n")














