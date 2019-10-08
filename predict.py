"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from tkinter import *
import tkinter as tk


warnings.filterwarnings("ignore")



def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, true data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '01-10-2006 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Count')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

#def predict(my_scats, my_day, st, et, file1, file2): 
#    test = pd.read_csv(file2)
#    my_scats = int(my_scats)
#    st = int(st)
#    et = int(et)      
#    my_day = int(my_day)
#
#    test = test[(test.SCATS == my_scats) & (test.DAY == my_day) 
#    & (test.TIME >= st) & (test.TIME <= et) ]
#        
#    test.to_csv('custom.csv', encoding='utf-8', index=False)
#    custom = 'custom.csv'
#    _, _, X_test, y_test, scaler = process_data(file1, custom, lag)
#    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
#    
#    y_preds = []
#    for name, model in zip(names, models):
#        if name == 'SAEs':
#            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
#        else:
#            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#    file = 'images/' + name + '.png'
#    plot_model(model, to_file=file, show_shapes=True)
#    predicted = model.predict(X_test)
#    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
#    y_preds.append(predicted[:96])
#    eva_regress(y_test, predicted)
#        
#    # Get average traffic count per hour
#    time_range = int((et - st) / 100)
#    count = 0
#    for i in range(time_range):
#        count += predicted[i]
#        
#    count_phour = count / time_range
#    
#    
#    return count_phour
    
def main():
    lstm = load_model('model/lstm8.h5')
  
    gru = load_model('model/gru8.h5')
   
    saes = load_model('model/saes8.h5')
  
    models = [lstm, gru, saes]
    names = ['LSTM', 'LSTM8', 'GRU', 'GRU8', 'SAEs', 'SAEs8']

    #lag = 96

    master = tk.Tk()
    tk.Label(master, text="Enter Scats: ").grid(row=0)
    tk.Label(master, text="Enter day 0=mon etc ").grid(row=2)
    tk.Label(master, text="Start time: ").grid(row=3)
    tk.Label(master, text="End time: ").grid(row=4)
    tk.Label(master, text="Direction: ").grid(row=5)
    tk.Label(master, text="Average Traffic Count /hr: ").grid(row=6)

    scats = tk.Entry(master)
    day = tk.Entry(master)
    start = tk.Entry(master)
    end = tk.Entry(master)
    dir = tk.Entry(master)
    display = tk.Entry(master)


    scats.grid(row=0, column=1)
    day.grid(row=2, column=1)
    start.grid(row=3, column=1)
    end.grid(row=4, column=1)
    dir.grid(row=5, column=1)
    display.grid(row=6, column=1)
    
    def predict():

        # Get input values
        my_scats = int(scats.get())
        st = int(start.get())
        et = int(end.get())
        my_day = int(day.get())
        d = dir.get()
        lag= 8
        # Filter test file on input values
        file1 = './data/train1.csv'
        file2 = './data/test1.csv'
        train = pd.read_csv(file1)
        test = pd.read_csv(file2)

        train = train[(train['SCATS'] == my_scats) & (train['DAY'] == my_day) 
        & (train['TIME'] >= st) & (train['TIME'] <= et) & (train[d] > 0)]
        
        test = test[(test['SCATS'] == my_scats) & (test['DAY'] == my_day) 
        & (test['TIME'] >= st) & (test['TIME'] <= et) & (test[d] > 0)]
        
        test.to_csv('./data/my_test.csv', encoding='utf-8', index=False)
        train.to_csv('./data/my_train.csv', encoding='utf-8', index=False)
        my_test = './data/my_test.csv'
        my_train = './data/my_train.csv'
        _, _, X_test, y_test, scaler = process_data(my_train, my_test, lag)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    
        y_preds = []
        for name, model in zip(names, models):
            if name == 'SAEs':
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            else:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:96])
        print(name)
        eva_regress(y_test, predicted)
        
        # Get average traffic count per hour
        time_range = int((et - st) / 100)
        count = 0
        for i in range(time_range):
            count += predicted[i]
        
        count_phour = count / time_range
        display.insert(0, count_phour)
        #plot_results(y_test[: 96], y_preds, names)
        #plot_results(y_test, y_preds, names)
    tk.Button(master, 
          text='Quit', 
          command=master.quit).grid(row=7, 
                                    column=0, 
                                    sticky=tk.W, 
                                    pady=4)
    tk.Button(master, 
          text='Predict', 
          command=predict).grid(row=7, 
                                    column=1, 
                                    sticky=tk.W, 
                                    pady=4)

#    tk.Button(master, text='Predict', command=predict).grid(row=6, column=1,sticky=tk.W,pady=4)
                                                            
#    count_phour = predict(my_scats, my_day, st, et, file1, file2)
#    display.insert(0, count_phour)
    master.mainloop()

    tk.mainloop()


if __name__ == '__main__':
    main()