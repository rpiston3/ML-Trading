from os import terminal_size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LSTM as L

def buy_and_hold(true_labels, starting_money):
    num_shares = starting_money/true_labels[0]
    end_money = num_shares * true_labels[-1]
    return end_money

#true labels should be the real value, predictions should be a percent change (1% not 0.01)
def trader_simple(predictions, true_labels, starting_money):
    port_value = starting_money
    current_action = 'not invested'
    wealth = []
    wealth.append(port_value)
    num_shares = 0
    for i in range(len(predictions) - 1):
        #if we predict we go up by at least .15% and not already invested and cant get free shares
        if (predictions[i+1] > 1) and (current_action == 'not invested') and (true_labels[i] != 0):
            #buy as many shares as we can (allow fractional shares)
            num_shares = port_value/true_labels[i]
            current_action = 'invested'
        #if we predict we go down and then we are already invested and we have shares to sell and dont sell shares for nothing
        elif (predictions[i+1] < -1) and (current_action == 'invested') and (num_shares != 0) and (true_labels[i] != 0):
            #sell shares and update new portfolio value and set action to not invested
            port_value = num_shares*true_labels[i]
            num_shares = 0
            current_action = 'not invested'
        else:
            #hold our shares
            if num_shares != 0 and true_labels[i] != 0 and (current_action == 'invested'):
                port_value = num_shares*true_labels[i]
        wealth.append(port_value)
    return wealth[-1]

#actual is real prices, predicted is pct change 
def buy_sell_trades(actual, predicted):
    money = 10000
    number_of_stocks = (int)(10000 / actual[0])
    left = 10000 - (int)(10000 / actual[0]) * actual[0] + actual[len(actual) - 1] * number_of_stocks

    number_of_stocks = 0

    buying_percentage_threshold = 0.15 #as long as we have a 0.15% increase/decrease we buy/sell the stock
    selling_percentage_threshold = 0.15

    for i in range(len(predicted) - 1):    
        if predicted[i + 1] > buying_percentage_threshold:
            for j in range(100, 0, -1):
                #Buying of stock
                if (money >= j * actual[i]):
                    money -= j * actual[i]
                    number_of_stocks += j
                    break
        elif  predicted[i + 1] < -selling_percentage_threshold:
            for j in range(100, 0, -1):
                #Selling of stock
                if (number_of_stocks >= j):
                    money += j * actual[i]
                    number_of_stocks -= j
                    break

    money += number_of_stocks * actual[len(actual) - 1]
    #print('function moneyy and then bh from function')
    #print(money) #Money if we traded
    #print(left)  #Money if we just bought as much at the start and sold near the end (Buy and hold)

    return money, left
#pass in actual prices and predicted prices not pct change
def compare_portfolios(true_labels, predictions, starting_money):
    #list to series find pct change then go back
    mySeries = pd.Series(predictions)
    pct_change = mySeries.pct_change()
    predicted = mySeries.tolist()
    #predicted = [100*x for x in predicted]
    #find values
    buy_hold_value = buy_and_hold(true_labels, starting_money)
    trader_value = trader_simple(predicted, true_labels, starting_money)
    github_value_end, github_value_holding = buy_sell_trades(true_labels, predicted)
    #store values 
    values = pd.DataFrame([[buy_hold_value, trader_value, github_value_end, github_value_holding]], columns=['buy and hold', 'trader value', 'github value end', 'github value holding'])
    values = values.round(2)
    return values
