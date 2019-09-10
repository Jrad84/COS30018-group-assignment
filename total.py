#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:54:47 2019

@author: aseem
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras.layers import Input, Dense, TimeDistributed, concatenate, LSTM
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
import pandas as pd
from pandas import Series
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import matplotlib as mp
    
# contains the total for each day (1 = sunday)
dataset_path = "./total.csv"
raw_dataset = pd.read_csv(dataset_path, header=None, index_col=False, 
                          names=['SCAT', 'street', 'Day', 'Date', 'Traffic'])
raw_dataset = raw_dataset[['SCAT', 'street', 'Day', 'Traffic']]

# all 3 input (x) variables are catagorical and output (y) variable numaric
raw_dataset['SCAT'] = raw_dataset.SCAT.astype('category')
raw_dataset['street'] = raw_dataset.street.astype('category')
raw_dataset['Day'] = raw_dataset.Day.astype('category')

'''
to check catagories of data
print(raw_dataset.SCAT.dtype)
print(raw_dataset.Day.dtype)
print(raw_dataset.street.dtype)
print(raw_dataset.Traffic.dtype)
'''
#raw_dataset information


# 40 SCAT
print(raw_dataset.SCAT.value_counts())
print(raw_dataset.SCAT.describe())
print("---------------48--------------")


# streets 139 
print(raw_dataset.street.value_counts())
print(raw_dataset.street.describe()) 
'''
street_file = open("street.txt")
street_file.write(str(print(raw_dataset.street.value_counts())))
street_file.close()
'''
print("-------------59----------------")


# Day some days missing
print(raw_dataset.Day.value_counts())
print(raw_dataset.Day.describe()) 
print("-------------65----------------")


#dataset = raw_dataset.iloc[:, :].values

#print("--------------70---------------")
#plt.scatter(raw_dataset['SCAT'], raw_dataset['Traffic'])
#sns.catplot(x='SCAT', y='Traffic', data=raw_dataset, kind='box')
#sns.catplot(x='Day', y='Traffic', data=raw_dataset, kind='box')

#print("--------------75---------------")

# ONE-HOT for Nominal variables - all 2 of them street and day
new_data = raw_dataset
dummies1 = pd.get_dummies(new_data.street)
new_data = pd.concat([new_data, dummies1], axis='columns')
dummies2 = pd.get_dummies(new_data.Day)
new_data = pd.concat([new_data, dummies2], axis='columns')

# drop street, day & drop 1 street and monday 
new_data = new_data.drop('street', axis='columns')
new_data = new_data.drop('Day', axis='columns')
new_data = new_data.drop('SCAT', axis='columns')
new_data = new_data.drop('CAMBERWELL_RD NW of BURKE_RD', axis='columns')
new_data = new_data.drop(1, axis='columns')

# divide data into X-input and Y-output
X = new_data.loc[:, new_data.columns != "Traffic"]
Y = new_data['Traffic']

from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2) # random_state=10






# Functional model
inputs = Input(shape=(144,))
# # a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# # This creates a model that includes
# # the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
print("--------------108---------------")

m = model.fit(
        X_train, Y_train,
        batch_size=256,
        epochs=600,
        validation_split=0.2)
model.save('nn.h5')
df = pd.DataFrame.from_dict(m.history)
df.to_csv('loss.csv', encoding='utf-8', index=False)

print("--------------121---------------")

s = Model('nn.h5')
plot_model(model, show_shapes=True)
predicted = model.predict(X_test)
#print(predicted)
Y_predicted = []
Y_predicted.append(predicted[:839])
print("--------------138---------------")

'''
#mape = MAPE(Y_test, Y_predicted)
y = [x for x in Y_test if x > 0]
yP = Y_predicted
num = len(Y_predicted)
sums = 0
for i in range(num):
    tmp = abs(y[i] - yP[i]) / y[i]
    sums += tmp

mape = sums * (100 / num)

#vs = metrics.explained_variance_score(Y_test, Y_predicted)
#mae = metrics.mean_absolute_error(Y_test, Y_predicted)
#mse = metrics.mean_squared_error(Y_test, Y_predicted)
#r2 = metrics.r2_score(Y_test, Y_predicted)
#print('explained_variance_score:%f' % vs)
#print("mape " + str(mape) + " ssss")
#print('mae:%f' % mae)
#print('mse:%f' % mse)
#print('rmse:%f' % math.sqrt(mse))
#print('r2:%f' % r2)
print("--------------151---------------")
'''

# Linear Regression
from sklearn.linear_model import LinearRegression as LR
clf = LR()
clf.fit(X_train, Y_train)
clf.predict(X_test)
#print(clf.predict(X_test))
print(clf.score(X_test, Y_test)) # get 0.98 very good
print(clf.score(X_train, Y_train))
