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

dataset_path = "./data/V2.csv"
# Drop rows with any empty cells
raw_dataset = pd.read_csv(dataset_path, header=None, index_col=False, 
                          names=['SCATS', 'DIRECTION', 'DATE', 'TIME', 'COUNT'])

    
raw_dataset['SCATS'] = raw_dataset.SCATS.astype('category')
raw_dataset['DIRECTION'] = raw_dataset.DIRECTION.astype('category')   

new_data = raw_dataset
dummies1 = pd.get_dummies(new_data.DIRECTION)
new_data = pd.concat([new_data, dummies1], axis='columns')  
new_data = new_data.drop('DIRECTION', axis='columns')

# divide data into X-input and Y-output
X = new_data.loc[:, new_data.columns != "COUNT"]
Y = new_data['COUNT']

from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2) # random_state=10


model = Sequential([
    Dense(32, activation='relu', input_shape=(7,)),
    Dense(1, activation='softmax'),
])
 
model.summary()
 
# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')
 
 
hist = model.fit(
     X_train, 
     Y_train, 
     batch_size=32, 
     epochs=100, verbose=2, 
     callbacks=None, 
     validation_split=0.2, 
     validation_data=None, 
     shuffle=True, 
     class_weight=None, 
     sample_weight=None, 
     initial_epoch=0)

model.save('nn.h5')
df = pd.DataFrame.from_dict(hist)
df.to_csv('loss.csv', encoding='utf-8', index=False)
predicted = model.predict(X_test)
# Functional model
#inputs = Input(shape=(9,))
## # a layer instance is callable on a tensor, and returns a tensor
#x = Dense(64, activation='relu')(inputs)
#x = Dense(64, activation='relu')(x)
#predictions = Dense(9, activation='sigmoid')(x)
# 
## # This creates a model that includes
## # the Input layer and three Dense layers
#model = Model(inputs=inputs, outputs=predictions)
#model.compile(optimizer='rmsprop',
#              loss='mse',
#              metrics=['mape'])
#
#hist = model.fit( 
#     X_train,
#     Y_train, 
#     batch_size=32, 
#     epochs=100, verbose=2, 
#     callbacks=None, 
#     validation_split=0.05, 
#     validation_data=None, 
#     shuffle=True, 
#     class_weight=None, 
#     sample_weight=None, 
#     initial_epoch=0)




    



