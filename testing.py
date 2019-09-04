# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:01:17 2019

@author: User
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.layers import Input, Dense, TimeDistributed, concatenate, LSTM
from keras.models import Model, Sequential
import pandas as pd
from pandas import Series
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime

column_names = ['NORTH', 'EAST', 'SOUTH', 'WEST', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN', '0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
times = ['0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
    

dataset_path = "./data/970.csv"
raw_dataset = pd.read_csv(dataset_path)
validate = pd.read_csv("./data/970_labels.csv")
vd = validate.iloc[:, :].values
dataset = raw_dataset.iloc[:, :].values
# encode text
transformer = ColumnTransformer(
        transformers=[
                ("OneHot",        # Just a name
                 OneHotEncoder(), # The transformer class
                 [0, 1]           # The column(s) to be applied on.
                 )
                ],
        remainder='passthrough' # donot apply anything to the remaining columns
        )

        
dataset = transformer.fit_transform(dataset.tolist())
vd = transformer.fit_transform(dataset.tolist())
my_data = pd.DataFrame(dataset)
labels = pd.DataFrame(vd)

        
# Split data into training 80% and testing 20%
train_dataset = my_data.sample(frac = 0.8, random_state = 0)
validate_train = labels.sample(frac = 0.8, random_state = 0)
test_dataset = my_data.drop(train_dataset.index)
validate_test = labels.drop(validate_train.index)
        
# View some stats 
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
        
# Scale data
x_scaler = MinMaxScaler()
train_scaled = x_scaler.fit_transform(train_dataset)
validate_scaled = x_scaler.fit_transform(validate_train)
test_scaled = x_scaler.fit_transform(test_dataset)

# =============================================================================
# model = Sequential([
#     Dense(32, activation='relu', input_shape=(107,)),
#     Dense(109, activation='softmax'),
# ])
# 
# model.summary()
# 
# # For a mean squared error regression problem
# model.compile(optimizer='rmsprop',
#               loss='mse')
# 
# 
# model.fit(
#      train_scaled, 
#      validate_train, 
#      batch_size=32, 
#      epochs=1000, verbose=2, 
#      callbacks=None, 
#      validation_split=0.2, 
#      validation_data=None, 
#      shuffle=True, 
#      class_weight=None, 
#      sample_weight=None, 
#      initial_epoch=0)
# =============================================================================

# Functional model
inputs = Input(shape=(107,))
# # a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(109, activation='sigmoid')(x)
 
# # This creates a model that includes
# # the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
model.fit(
     #train_dataset, 
     train_scaled,
     validate_scaled, 
     batch_size=32, 
     epochs=100, verbose=2, 
     callbacks=None, 
     validation_split=0.2, 
     validation_data=None, 
     shuffle=True, 
     class_weight=None, 
     sample_weight=None, 
     initial_epoch=0)
    

fig = plt.figure()  # an empty figure with no axes
fig.suptitle('Predicted Traffic Flow SCATS 970')  

fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes