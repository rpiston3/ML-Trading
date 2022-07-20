from fileinput import close
from pickletools import optimize
from random import shuffle
from unicodedata import name
from pandas.core.algorithms import diff
from pandas.core.frame import DataFrame
import data_grab as dg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras 
import keras_tuner
from keras_tuner import RandomSearch

def prep_train_data(strategy, data):
    if (strategy == 'darse') or (strategy == 'close'):
        close_price = pd.DataFrame()
        close_price['change'] = data['close']
        close_price = close_price.diff()
        close_price.replace([np.inf, -np.inf], np.nan, inplace=True)
        close_price = close_price.dropna()
        close_price['close'] = data['close']
        close_price = close_price[['close', 'change']]
        train, test = train_test_split(close_price, test_size=.2,shuffle=False)
    else:
        train, test = train_test_split(data, test_size=.2,shuffle=False)
    scalar = MinMaxScaler()
    train_scaled = scalar.fit_transform(train)
    ################### TRAIN ######################
    x_train = []
    y_train = []
    for i in range(21, train.shape[0]):
        x_train.append(train_scaled[i-21: i])
        #can be [i][0] or [i,0]
        y_train.append(train_scaled[i][0])
    x_all = np.array(x_train)
    y_all = np.array(y_train)
    xtrain, xval = train_test_split(x_train, test_size = .2, shuffle=False)
    ytrain, yval = train_test_split(y_train, test_size = .2, shuffle=False)
    x_train = np.array(xtrain)
    x_val = np.array(xval)
    y_train = np.array(ytrain)
    y_val = np.array(yval)

    return x_train, x_val, y_val, y_train, x_all, y_all


def get_predictions(train, test, scalar, strategy):
    ####################### TEST #########################################
    #take last 21 days of train and then add all of test so we can have all our values be defined for test set
    one_lookback = train.tail(21)
    dt = one_lookback.append(test, ignore_index = True)
    #scale all input data
    inputs = scalar.fit_transform(dt)
    #test set
    x_test = []
    y_test = []
    #append windows of length 21 (so each append we append an entire array with the last 21 inputs/dataframe inputs and each iteration we slide the window up one)
    for i in range(21, inputs.shape[0]):
        #append array from -21 to 0 first run then slide to -20 to 1 second run etc (sliding window) (i is exclusive)
        x_test.append(inputs[i-21:i])
        #append just the actual price for the ith value we want to predict
        y_test.append(inputs[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    #predict the prices
    y_pred = model.predict(x_test)
    scale = 1/scalar.scale_[0]
    #unscale to get true prices not scaled prices
    y_pred = y_pred*scale
    y_test = y_test*scale
    ############################ Choose correct output based on strategy ######################################
    #predict n days ahead based off closing price difference between today and yesterday and add difference to real results
    if strategy == 'darse':
        y_pred_diff = []
        for i in range(len(y_pred) - 1):
            differ = y_pred[i+1].mean() - y_pred[i].mean()
            y_pred_diff.append(differ)
        y_real_out = []
        for i in range(len(y_test)):
            y_real_out.append(y_test[i])
        y_pred_diff = np.array(y_pred_diff)
        y_real_out = np.array(y_real_out)
        y_pred_diff = np.insert(y_pred_diff, 0, 0)
        y_pred_out = y_real_out + y_pred_diff
        y_predictions = y_pred_out
        y_actual = y_test
    #predict 1 day ahead off all fetures
    elif strategy == 'full':
        yy = []
        for i in range(len(y_pred)):
           yy.append(y_pred[i][0])
        y_predictions = yy
        y_actual = y_test
    #compute n days ahead based off all features
    elif strategy == 'dense':
        output = []
        for i in range(len(y_pred)):
            output.append(y_pred[i].mean()) 
        y_predictions = output
        y_actual = y_test
    #n days based off of only closing price, but we dont take the difference and add it to real results
    elif strategy == 'close':
        output = []
        for i in range(len(y_pred)):
            output.append(y_pred[i].mean()) 
        y_predictions = output
        y_actual = y_test
    #take n days ahead based off all feats and add the difference to real results
    elif strategy == 'all feat difference results':
        y_pred_diff = []
        for i in range(len(y_pred) - 1):
            differ = y_pred[i+1].mean() - y_pred[i].mean()
            y_pred_diff.append(differ)
        y_real_out = []
        for i in range(len(y_test)):
            y_real_out.append(y_test[i])
        y_pred_diff = np.array(y_pred_diff)
        y_real_out = np.array(y_real_out)
        y_pred_diff = np.insert(y_pred_diff, 0, 0)
        y_pred_out = y_real_out + y_pred_diff
        y_predictions = y_pred_out
        y_actual = y_test
    # return list of predicted prices and list of actual prices not percentages
    return y_predictions, y_actual

def build_model(hp):
    ####################### BUILD MODEL ARCHITECTURE #####################################
    # units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
    # units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
    # units_3 = hp.Int("units_3", min_value=2, max_value=128, step=2)
    learning_rate = hp.Float("lr", min_value=.0001, max_value=.001, step=.0001)
    activation_1 = hp.Choice("activation_1", ["selu", "tanh"])
    activation_2 = hp.Choice("activation_2", ["selu", "tanh"])
    drop_1 = hp.Float("drop_1", min_value=.1, max_value=.5, step=.05)
    drop_2 = hp.Float("drop_2", min_value=.1, max_value=.5, step=.05)
    model = Sequential()
    model.add(LSTM(units = 256, activation = activation_1, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(rate = drop_1))
    model.add(LSTM(units = 256, activation = activation_2))
    model.add(Dropout(rate = drop_2))
    model.add(Dense(units = 256))
    model.add(Dense(units = days_ahead))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = opt, loss = 'mean_squared_error')
    return model

#original LSTM model
# def build_model(hp):
#     ####################### BUILD MODEL ARCHITECTURE #####################################
#     #units_1 = hp.Int("units_1", min_value=32, max_value=512, step=32)
#     #units_2 = hp.Int("units_2", min_value=32, max_value=512, step=32)
#     #units_3 = hp.Int("units_3", min_value=32, max_value=512, step=32)
#     #units_4 = hp.Int("units_4", min_value=32, max_value=512, step=32)
#     learning_rate = hp.Float("lr", min_value=.0001, max_value=.001, step=.0001)
#     activation_1 = hp.Choice("activation_1", ["selu", "tanh"])
#     activation_2 = hp.Choice("activation_2", ["selu", "tanh"])
#     activation_3 = hp.Choice("activation_3", ["selu", "tanh"])
#     activation_4 = hp.Choice("activation_4", ["selu", "tanh"])
#     drop_1 = hp.Float("drop_1", min_value=.1, max_value=.5, step=.05)
#     drop_2 = hp.Float("drop_2", min_value=.1, max_value=.5, step=.05)
#     drop_3 = hp.Float("drop_3", min_value=.1, max_value=.5, step=.05)
#     drop_4 = hp.Float("drop_4", min_value=.1, max_value=.5, step=.05)
#     model = Sequential()
#     model.add(LSTM(units = 60, activation = activation_1, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
#     model.add(Dropout(rate = drop_1))
#     model.add(LSTM(units = 70, activation = activation_2, return_sequences = True))
#     model.add(Dropout(rate = drop_2))
#     model.add(LSTM(units = 80, activation = activation_3, return_sequences = True))
#     model.add(Dropout(rate = drop_3))
#     model.add(LSTM(units = 90, activation = activation_4))
#     model.add(Dropout(rate = drop_4))
#     model.add(Dense(units = days_ahead))
#     model.summary()
#     opt = keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer = opt, loss = 'mean_squared_error')
#     return model

#could put all below in one functions and do days ahead in model to just be 1


data = dg.get_data('GME')
print(data)
x, vx, vy, y, x_all, y_all = prep_train_data('all feat difference results', data)
x_train = x
x_val = vx
y_train = y
y_val = vy
days_ahead = 1 



tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_loss",
        max_trials=2,
        executions_per_trial=2,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )
tuner.search_space_summary()
tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))

print(1)
best_hp = tuner.get_best_hyperparameters()[0]
print(2)
model = tuner.hypermodel.build(best_hp)
print(3)
summary = tuner.results_summary(num_trials=10)
print(4)
