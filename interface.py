import kivy 
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.tabbedpanel import TabbedPanel
import networkx as nx
import geopandas
import contextily as ctx 
from fiona.crs import from_epsg
from kivy.garden.matplotlib import FigureCanvasKivy, FigureCanvasKivyAgg
import seaborn
from kivy.uix.popup import Popup
import matplotlib.pyplot as plt
# from dfguik import DfguiWidget
import Map
import DrawMap

# import math
# import warnings
# import numpy as np
# import pandas as pd
# from data.data import process_data
# from keras.models import load_model
# from keras.utils.vis_utils import plot_model
# import sklearn.metrics as metrics
# import matplotlib as mpl
# import matplotlib.pyplot as plt


# class Menu(Widget):
#     startScats = ObjectProperty(None)
#     endScats = ObjectProperty(None)
#     day = ObjectProperty(None)
#     startTime = ObjectProperty(None)
#     endTime = ObjectProperty(None)
#     direction = ObjectProperty(None)
#     prediction = ObjectProperty(None)
    
    
#     def btn(self):

#         print("Scats: ", self.scatStart.text, "Day: ", self.day.text, "Start Time: ", self.startTime.text, "End time: ", self.endTime.text, "Direction: ", self.direction.text, "Prediction: ", self.prediction.text)
#         my_scats = np.int64(self.scatStart.text)
#         st = np.int64(self.startTime.text)
#         et = np.int64(self.endTime.text)
#         my_day = np.int64(self.day.text)
#         d = self.direction.text
#         self.predict(my_scats, st, et, my_day, d)

#     def getRoute(self):
#         route = Map.createRoute(np.int64(self.startScats.text), np.int64(self.endScats.text))
#         self.prediction.text = str(route)

#     lstm = load_model('model/lstm8.h5')
#     gru = load_model('model/gru8.h5')
#     saes = load_model('model/saes8.h5')
  
#     models = [lstm, gru, saes]
#     names = ['LSTM', 'LSTM8', 'GRU', 'GRU8', 'SAEs', 'SAEs8']

    
#     def predict(self, my_scats, st, et, my_day, d):

#         # Get input values
#         # my_scats = np.int64(self.scats.text)
#         # st = np.int64(self.startTime.text)
#         # et = np.int64(self.endTime.text)
#         # my_day = np.int64(self.day.text)
#         # d = self.direction.text

#         lag= 8
#         # Filter test file on input values
#         file1 = './data/train1.csv'
#         file2 = './data/test1.csv'
#         train = pd.read_csv(file1)
#         test = pd.read_csv(file2)

#         train = train[(train['SCATS'] == my_scats) & (train['DAY'] == my_day) 
#         & (train['TIME'] >= st) & (train['TIME'] <= et) & (train[d] > 0)]
        
#         test = test[(test['SCATS'] == my_scats) & (test['DAY'] == my_day) 
#         & (test['TIME'] >= st) & (test['TIME'] <= et) & (test[d] > 0)]
        
#         test.to_csv('./data/my_test.csv', encoding='utf-8', index=False)
#         train.to_csv('./data/my_train.csv', encoding='utf-8', index=False)
#         my_test = './data/my_test.csv'
#         my_train = './data/my_train.csv'
#         _, _, X_test, y_test, scaler = process_data(my_train, my_test, lag)
#         y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
    
#         y_preds = []
#         for name, model in zip(self.names, self.models):
#             if name == 'SAEs':
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
#             else:
#                 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
#         file = 'images/' + name + '.png'
#         plot_model(model, to_file=file, show_shapes=True)
#         predicted = model.predict(X_test)
#         predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
#         y_preds.append(predicted[:96])
#         print(name)
#         eva_regress(y_test, predicted)
        
#         # Get average traffic count per hour
#         time_range = int((et - st) / 100)
#         count = 0
#         for i in range(time_range):
#             count += predicted[i]
        
#         count_phour = count / time_range
#         self.prediction.text = str(count_phour)
#         #plot_results(y_test[: 96], y_preds, names)
#         #plot_results(y_test, y_preds, names)

class testApp(App):
    def build(self):
        data = DrawMap.DataFrame()
        box = BoxLayout(orientation="horizontal")

        # seaborn.set_palette('bright')
        # seaborn.set_style('whitegrid')
        # seaborn.pairplot(data=df,
        #                  hue="Point",
        #                  kind="scatter",
        #                  diag_kind="hist",
        #                  x_vars=("Latitude"),
        #                  y_vars=("Longitude"))

        # box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        # G=nx.Graph()
        # for index, row in data.iterrows():
        #     G.add_node(row.Point, y = row.Latitude , x = row.Longitude, 
        #             geometry=(row.Longitude, row.Latitude))
        
        # G = DrawMap.WithEdge(G)
        # print(G.nodes)
        route = Map.createRoute('970', '3682')
        print(route[0])
        plt = Map.CreateInitialMap(data)
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        # box.add_widget(FigureCanvasKivyAgg(route[1].gcf()))
        # box.add_widget(FigureCanvasKivy(plt.gcf()))
        return box

        
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


if __name__ == "__main__":
    testApp().run()