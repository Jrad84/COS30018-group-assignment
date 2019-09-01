# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:17:06 2019

@author: JT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, TimeDistributed, concatenate, LSTM
from keras.models import Model
from keras.utils import to_categorical
import pandas as pd
from pandas import Series
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime

class Data:
    def __init__(self):
        self.train = None
        self.test = None
        self.train_labels = None
        self.test_labels = None
        self.times = None
    
    def load_data(self):
        dataset_path = "./data/970_SUN_V1.csv"
        raw_dataset = pd.read_csv(dataset_path)
        #raw_dataset = pd.read_csv(dataset_path, names=column_names)
        dataset = raw_dataset.iloc[:, :].values
        # encode text
        transformer = ColumnTransformer(
                transformers=[
                        ("OneHot",        # Just a name
                         OneHotEncoder(), # The transformer class
                         [0]              # The column(s) to be applied on.
                         )
                        ],
                remainder='passthrough' # donot apply anything to the remaining columns
                )
        dataset = transformer.fit_transform(dataset.tolist())
        my_data = pd.DataFrame(dataset)
        self.times = ['0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
        self.train_labels = ['NORTH', 'EAST', 'SOUTH', 'WEST', '0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
        self.test_labels = ['LOCATION',  '0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45']
        # Split data into training - 80% and testing 20%
        train_dataset = my_data.sample(frac = 0.8, random_state = 0)
        test_dataset = my_data.drop(train_dataset.index)
        # View some stats 
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        # Scale data
        x_scaler = MinMaxScaler()
        train_scaled = x_scaler.fit_transform(train_dataset)
        test_scaled = x_scaler.fit_transform(test_dataset)

        self.train = pd.DataFrame(train_scaled)
        self.test = pd.DataFrame(test_scaled)
    
        return self.train, self.test, self.train_labels, self.test_labels, self.times




class Model():
    #Build the model (Functional) data_length = 96 (24 x 15), label_length = time period required for prediction
    def build_model(data_length, label_length):
        # first input layer is location
        location = Input(shape = (data_length, 1), name = 'location')
        # iterate through times creating a new input layer for each time
        for time in times:
            time = Input(shape = (data_length, 1), name = time)
    
        location_layers = LSTM(64, return_sequences=False)(location)
        time_layers = LSTM(64, return_sequences=False)(time)
        
        output = concatenate(
                [
                        location_layers,
                        time_layers,
                ]
            )
    
        output = Dense(label_length, activation='linear', name='weighted_avg_output')(output)
    
        model = Model(
                inputs = 
                [
                        location,
                        time
                ],
                outputs =
                [
                        output
                ]
                )
    
        model.compile(optimizer='rmsprop', loss='mse')
    
        return model
    
if __name__ == '__main__':
    my_data = Data()
    model = Model()
    train_data = my_data.train
    test_data = my_data.test
    train_labels = my_data.train_labels
    test_labels = my_data.test_labels
    times = my_data.times
    rnn = model.build_model(96, 96)
    
    rnn.fit(
        [
                train_data["location"],
                train_data[times],
        ],
        [

                #for time in times:
                train_labels[times]
        ],
        validation_data=(
                [
                        test_data["location"],
                        test_data[times],
                ],
                [
                        #for time in times:
                        test_labels[times]
                ]),
        epochs = 1,
        batch_size = 3000,
        callbacks = [
                # Create tensorboard to view model / training
                #board.createTensorboardConfig('logs/Graph'),
        ])





