"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
#from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
#import matplotlib as mpl
#import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

class CleanPrediction():
    def __init__(self, **kwargs):
        lstm = load_model('model/lstm4_layers_4.h5')
        gru = load_model('model/gru4_layers_4.h5')
   
        saes = load_model('model/saes4_layers_4.h5')

        simple_rnn = load_model('model/simplernn4_layers_4.h5')

        self.models = [lstm, gru, saes, simple_rnn]
        self.names = ['LSTM', 'GRU', 'SAEs', 'SIMPLE_RNN']
        # self.run()
        
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


    def eva_regress(self, y_true, y_pred):
        """Evaluation
        evaluate the predicted resul.
        # Arguments
            y_true: List/ndarray, ture data.
            y_pred: List/ndarray, predicted data.
            """
        eval = []
        mape = round(CleanPrediction.MAPE(y_true, y_pred),2)
        vs = metrics.explained_variance_score(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
#        print('explained_variance_score:%f' % vs)
#        print('mape:%f%%' % mape)
#        print('mae:%f' % mae)
#        print('mse:%f' % mse)
#        print('rmse:%f' % math.sqrt(mse))
#        print('r2:%f' % r2)
        eval = mape, vs, mae, mse, r2
        return eval

 
        
    def predict(self, my_scats, st, et, my_day, d):
        print(my_scats, st, et, my_day, d)
        lag= 4
        # Filter test file on input values
        file1 = './data/train1.csv'
        file2 = './data/test1.csv'
        train = pd.read_csv(file1)
        test = pd.read_csv(file2)
        error = 0

        train = train[(train['SCATS'] == my_scats) & (train['DAY'] == my_day) 
        & (train['TIME'] >= st) & (train['TIME'] <= et) & (train[d] > 0)]
        
        test = test[(test['SCATS'] == my_scats) & (test['DAY'] == my_day) 
        & (test['TIME'] >= st) & (test['TIME'] <= et) & (test[d] > 0)]
        
        # If test data <= lag, return error
        if (len(test.index) <= 4):
            print(len(test.index))
            return error
        
        else:
            test.to_csv('./data/my_test.csv', encoding='utf-8', index=False)
            train.to_csv('./data/my_train.csv', encoding='utf-8', index=False)
            my_test = './data/my_test.csv'
            my_train = './data/my_train.csv'
            _, _, X_test, y_test, scaler = process_data(my_train, my_test, lag)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

            y_preds = []
            for name, model in zip(self.names, self.models):
                if name == 'SIMPLE_RNN':
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
                else:
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
 
            # plot_model(model, to_file=file, show_shapes=True)
            predicted = model.predict(X_test)
            predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
            y_preds.append(predicted[:96])
            metrics = self.eva_regress(y_test, predicted)
        
            # Get average traffic count per hour
            time_range = int((et - st) / 100)
            count = 0
            
            # Check predictions, if only 1 then return it
            if (len(predicted) == 1):
                count = predicted[0]
            else:
                for i in range(time_range):
                    count += predicted[i]
        
            count_phour = np.int64(round((count / time_range),0))
            print(count_phour)
            return count_phour, metrics
     