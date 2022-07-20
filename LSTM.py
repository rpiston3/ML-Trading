from fileinput import close
from pickletools import optimize
from pandas.core.algorithms import diff
from pandas.core.frame import DataFrame
import data_grab as dg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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



#strategy can be == sparse | full | dense (t+n)
#data is dg.tsla_df_ind
#days ahead must be integer
#############CHANGE PARAMS TO INCLUDE STUFF IF WE HAVE 4 LAYERS#####################
def lstm_model(strategy, data, days_ahead, stock, u1, u2, u3, u4, lr, a1, a2, a3, a4, d1, d2, d3, d4):
    #sparse preprocessing to give split the right input
    #if darse then we use difference in closing price to learn
    if (strategy == 'darse') or (strategy == 'close'):
        close_price = pd.DataFrame()
        close_price['change'] = data['close']
        close_price = close_price.diff()
        close_price.replace([np.inf, -np.inf], np.nan, inplace=True)
        close_price = close_price.dropna()
        close_price['close'] = data['close']
        close_price = close_price[['close', 'change']]
        train, test = train_test_split(close_price, test_size=.2,shuffle=False)
    #split data and scale data
    else:
        train, test = train_test_split(data, test_size=.2,shuffle=False)
    print('train datagrame')
    print(train)
    scalar = MinMaxScaler()
    train_scaled = scalar.fit_transform(train)
    #test_scaled = scalar.fit_transform(test)
    ################### TRAIN ######################
    #split training into X_train and Y_train (x_train is indicators y train is price/predicted price)
    x_train = []
    y_train = []
    #define lookback period to be 21, from rows 21 to end
    for i in range(21, train.shape[0]):
        #get rows i-21 to i [,) i.e i is exclusive on right end
        #use last 21 days for the x variable
        x_train.append(train_scaled[i-21: i])
        #in row i get the first col value ie price (-1 is % cahnge and 0 is price)
        #use the next day for the y variable
        y_train.append(train_scaled[i, 0])
        #SO WE use these x train rows to predict the y train row
    #turn into arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #shape of x_train = (number of total data points, lookback period, number features),  y = (number of data points)
    ####################### BUILD MODEL ARCHITECTURE #####################################
    model = Sequential()
    #shape 1 = lookback period, shape 2 is number of features (input is (lookback period, num features)
    model.add(LSTM(units = 256, activation = a1, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(d1))
    model.add(LSTM(units = 256, activation = a2))
    model.add(Dropout(d2))
    model.add(Dense(units= 256))

    #model.add(LSTM(units = 60, activation = 'tanh', return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
    #model.add(Dropout(d1))
    #model.add(LSTM(units = 70, activation = 'tanh', return_sequences = True))
    #model.add(Dropout(d2))
    #model.add(LSTM(units = 80, activation = 'tanh', return_sequences = True))
    #model.add(Dropout(d3))
    #model.add(LSTM(units = 90, activation = 'tanh'))
    #model.add(Dropout(d4))






    #dense output is array of arrays of size|units| ex) units = 2 then [[7, 4], [9, 8], ..... [10, 11]]
    if strategy != 'full':
        model.add(Dense(units = days_ahead))
    else:
        model.add(Dense(units = 1))
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer = opt, loss = 'mean_squared_error')
    print("learning rate")
    print(model.optimizer.learning_rate)
    history = model.fit(x_train, y_train, epochs=1, batch_size = 100, validation_split = 0.3)
    

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
        #use the last 21 days as input
        x_test.append(inputs[i-21:i])
        #append just the actual price for the ith value we want to predict
        #use the next day as output
        y_test.append(inputs[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    #predict the prices
    y_pred = model.predict(x_test)
    #compute what to undo scaling by 
    scale = 1/scalar.scale_[0]
    y_pred = y_pred*scale
    y_test = y_test*scale
    print('check to see if training is lined up right')
    tt = x_train*scale
    ty = y_train*scale
    print('xtrain')
    print(tt[0])
    print('ytrain')
    print(ty[22])
    print('xtrain')
    print(tt[23])
    print('ytrain')
    print(ty[23])
    print(tt[24])
    ############################ Choose correct output based on strategy ######################################
    #predict n days ahead based off closing price difference between today and yesterday and add difference to real results
    if strategy == 'darse':
        y_pred_diff = []
        # # # # # # # # #append predictions from df to list
        for i in range(len(y_pred) - 1):
            differ = y_pred[i+1].mean() - y_pred[i].mean()
            y_pred_diff.append(differ)
        y_real_out = []
        # # # # # # # # #append real results from dataframe to list
        for i in range(len(y_test)):
            y_real_out.append(y_test[i])
        y_pred_diff = np.array(y_pred_diff)
        # # # # # # # # #actual prices 
        y_real_out = np.array(y_real_out)
        y_pred_diff = np.insert(y_pred_diff, 0, 0)
        # # # # # # # # #predicted prices when we add predicted change to the actual prices
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
    return y_predictions, y_actual, history

def perf_metrics(y_predictions, y_actual, og_data, ticker):
    #create 2 dataframes with percentage change for each price point for pred and test
    y_pred_change = pd.DataFrame(y_predictions, columns=['percent change pred'])
    y_test_change = pd.DataFrame(y_actual, columns=['percent change test'])
    y_pred_change = y_pred_change.pct_change()
    y_test_change = y_test_change.pct_change()
    y_pred_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_test_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_pred_change = y_pred_change.dropna()
    y_test_change = y_test_change.dropna()
    #multiply 100 to get pct (so 0.0015 becomes .15% or 0.10 becomes 10%)
    y_pred_change['percent change pred'] = 100*y_pred_change['percent change pred']
    y_test_change['percent change test'] = 100*y_test_change['percent change test']
    y_pred_change = y_pred_change.astype({'percent change pred': float})
    y_test_change = y_test_change.astype({'percent change test': float})
    frames = [y_pred_change, y_test_change]
    #dataframe that hold %change in price of predicted values and test values in their own columns
    result = pd.concat(frames, axis=1, join='inner')
    total_days = 0
    num_days_up = 0
    num_days_down = 0
    correct_days_up = 0
    correct_days_down = 0
    total_correct = 0
    up_days = pd.DataFrame(columns= ['pct change pred', 'pct change test'])
    down_days = pd.DataFrame(columns= ['pct change pred', 'pct change test'])
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    fake1 = 0
    fake2 = 0
    for index, row in result.iterrows():
        #if actual day is up add to list of up days and count 
        if row['percent change test'] > 0:
            up_days = up_days.append({'pct change pred' : row['percent change pred'], 'pct change test' : row['percent change test']}, ignore_index=True)
            num_days_up = num_days_up + 1
        #if actual day is down add to list of down days and count
        if row['percent change test'] < 0:
            down_days = down_days.append({'pct change pred' : row['percent change pred'], 'pct change test' : row['percent change test']}, ignore_index=True)
            num_days_down = num_days_down + 1
        #if predict and actual are both up add to correct up days and total correct
        if row['percent change test'] > 0 and row['percent change pred'] > 0:
            correct_days_up = correct_days_up + 1
            total_correct = total_correct + 1
            tp = tp + 1
        #if predict and actualy are both down add to correct down days and total correct
        elif row['percent change test'] < 0 and row['percent change pred'] < 0:
            correct_days_down = correct_days_down + 1
            total_correct = total_correct + 1
            tn = tn + 1
        elif row['percent change test'] < 0 and row['percent change pred'] > 0:
            fp = fp + 1
        elif row['percent change test'] > 0 and row['percent change pred'] < 0:
            fn = fn + 1
        else:
            print('real')
            print(row['percent change test'])
            print('pred')
            print(row['percent change pred'])
            fake1 = fake1 + 1
        total_days = total_days + 1
    pct_total_correct = (total_correct/total_days)*100
    pct_up_correct = (correct_days_up/num_days_up)*100
    pct_down_correct = (correct_days_down/num_days_down)*100
    ################# COMPUTE CORRELATIONS #####################
    corr_up = up_days.corr()
    corr_down = down_days.corr() 
    #corr between pct change in predict vs actual   
    correlation = result.corr()
    #see correlation of features before we train 
    indicators_corr = og_data.corr()
    plt.figure(figsize=[28,12])
    g = sns.heatmap(indicators_corr, annot=True)
    plt.savefig("{}_heatmap.png".format(ticker))
    ############################## ERROR CALCULATIONS ##############################
    #calcualte error between the percatange change in prices not the raw pricing (1% not 0.01)
    actual = result['percent change test'].to_numpy()
    predictions = result['percent change pred'].to_numpy()
    #MAe
    mae = mean_absolute_error(actual, predictions)
    #mse
    mse = mean_squared_error(actual, predictions)
    #RMSE
    rmse = mean_squared_error(actual, predictions, squared=False)
    #return correlations between pct change, and then pct correct and then the errors of predicted pct change and actual pct change
    col_list = []
    row_list = []
    col_list.append('pct total correct')
    col_list.append('pct up correct')
    col_list.append('pct down correct')
    col_list.append('mae')
    col_list.append('mse')
    col_list.append('rmse')
    row_list.append(pct_total_correct)
    row_list.append(pct_up_correct)
    row_list.append(pct_down_correct)
    row_list.append(mae)
    row_list.append(mse)
    row_list.append(rmse)
    values = pd.DataFrame([[pct_total_correct, pct_up_correct, pct_down_correct, mae, mse, rmse]], columns=col_list)
    return values, correlation, corr_up, corr_down, tp, fp, tn, fn, total_days, fake1, fake2

def graph_LSTM_results(y_actual, y_predictions, ticker, strategy):
    plt.clf()
    plt.plot(y_actual, color = 'red', label = 'Real {} Price'.format(ticker))
    plt.plot(y_predictions, color = 'blue', label = 'Predicted {} Price'.format(ticker))
    plt.title('{} Price Prediction For {} Strategy'.format(ticker, strategy))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('{}_{}.png'.format(ticker, strategy))

def graph_val(stock, strategy, history):
    #plt.rcParams["figure.figsize"] = (6.4, 4.8)
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('{}_{}_loss.png'.format(stock, strategy))

def pretty_printer(corr, up, down, performance, tp, fp, tn, fn, total, fake1, fake2):
    print("Now printing the following values, Percent total days predicted correct, Percent up days predicted correct, Percent down days predicted correct, MAE, MSE, RMSE")
    print(performance)
    print("Printing Correlation between predicted labels and actual labels for all days")
    print(corr)
    print("Printing Correlation between predicted labels and actual labels for up days")
    print(up)
    print("Printing Correlation between predicted labels and actual labels for down days")
    print(down)
    print('Accuracy (all correct over all')
    print((tp+tn)/(tp+tn+fp+fn))
    print('Misclassification (all incorrect / all)')
    print((fp+fn)/(tp+tn+fp+fn))
    print('Precision (true positives / predicted positives)')
    print(tp/(tp+fp))
    print("Sensitivity aka Recall (true positives / all actual positives)")
    print(tp/(tp+fn))
    print('Specificity (true negatives / all actual negatives)')
    print(tn/(tn+fp))
    print('check that total is same as all')
    print(total)
    print(tp+fp+tn+fn)
    print(fake1)
    print(fake2)
    #print("Money made accross buy and hold, our algo, and the guthub function with threshhold")
    #print(money)

#print('Total dataset has {} datapoints and {} features'.format(dg.tsla_df_ind.shape[0], dg.tsla_df_ind.shape[1]))
#print('Total dataset has {} datapoints and {} features'.format(test.shape[0], test.shape[1]))

