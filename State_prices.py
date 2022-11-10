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

"""
Values to customized:
set risk-free rate as 1.1
set initial share price as 45
set option_to_buy_A_B as 100.

final_share price:
State        Good   Normal   Bad
Phys Prob    0.3    0.5      0.2
Stock A      75     55       20
Stock B      60     50       40

"""
probability_good = 0.3
probability_normal = 0.5
probability_bad = 0.2
stockA_good = 75
stockA_normal = 55
stockA_bad = 20
stockB_good = 60
stockB_normal = 50
stockB_bad = 40
risk_free_rate = 1.1
initial_share_price = 45
option_to_buy_A_B = 100

final_share_price = [[probability_good, probability_normal, probability_bad], 
    [stockA_good, stockA_normal, stockA_bad],
    [stockB_good, stockB_normal, stockB_bad]]
final_share_price = np.array(final_share_price)



"""


Vector of state prices:
[1/1.1 45 45] * [1 75 60
                1 55 50
                1 20 40]^-1

= [0.2273 0.4091 0.2727]


"""

#vector of state prices
oneXthree_matrix = [[1/risk_free_rate,initial_share_price,initial_share_price]]
oneXthree_matrix = np.array(oneXthree_matrix)

threeXthree_matrix = [[1, stockA_good, stockB_good], 
    [1, stockA_normal, stockB_normal],
    [1, stockA_bad, stockB_bad]]

threeXthree_matrix = np.array(threeXthree_matrix)
threeXthree_matrix_inverse = np.linalg.inv(threeXthree_matrix)
vector_of_state_prices = np.matmul(oneXthree_matrix,threeXthree_matrix_inverse)


"""
Vector of risk-neutral probabilities:
[0.2273 0.4091 0.2727] * 1.1 = [0.25 0.45 0.30]

"""

vector_of_risk_neutral_probabilities = vector_of_state_prices*risk_free_rate
#print(vector_of_state_prices)
#print(vector_of_risk_neutral_probabilities)


"""
Payoff vector:
[max 75 + 60 − 100, 0      [35
 max 55 + 50 − 100, 0    =  5
 max 20 + 40 − 100, 0]      0]

"""

pay_off_vector = [max(threeXthree_matrix[0,1]+threeXthree_matrix[0,2]-option_to_buy_A_B,0), 
    max(threeXthree_matrix[1,1]+threeXthree_matrix[1,2]-option_to_buy_A_B,0),
    max(threeXthree_matrix[2,1]+threeXthree_matrix[2,2]-option_to_buy_A_B,0)]
pay_off_vector = np.array(pay_off_vector)

"""
Call price:
35 × 0.2273 + 5 × 0.4091 = 10

"""

call_price = pay_off_vector[0] * vector_of_state_prices[0,0] + pay_off_vector[1] * \
    vector_of_state_prices[0,1] + pay_off_vector[2] * vector_of_state_prices[0,2]


print("vector_of_state_prices: ", vector_of_state_prices, "\n")
print("vector_of_risk_neutral_probabilities: ", vector_of_risk_neutral_probabilities, "\n")
print("pay_off_vector:", pay_off_vector, "\n")
print("call_price:", call_price, "\n")














