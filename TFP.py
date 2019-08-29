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
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Import data
#dataset = pd.read_csv("./data/scats-data-V2.csv")
#X = dataset.iloc[:,:].values
#Y = pd.DataFrame(X)
dataset_path = "./data/970_SUN.csv"
column_names = ['SCATS', 'LOCATION', 'LATITUDE', 'LONGITUDE', 'DAY', '0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45']
raw_dataset = pd.read_csv(dataset_path, names=column_names)
raw_dataset.pop('DAY')
dataset = raw_dataset.iloc[:, :].values


# encode text
# =============================================================================
# label_encoder = LabelEncoder()
# dataset[:, 1] = label_encoder.fit_transform(dataset[:, 1])
# =============================================================================

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
dataset = transformer.fit_transform(dataset.tolist())
my_data = pd.DataFrame(dataset)


# Split data into training - 80% and testing 20%

train_dataset = my_data.sample(frac = 0.8, random_state = 0)
test_dataset = my_data.drop(train_dataset.index)
# # 
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

# =============================================================================
# train_labels = train_dataset.pop('0:00')
# test_labels = test_dataset.pop('0:00')
# =============================================================================

# =============================================================================
# def norm(x):
#   return (x - train_stats['mean']) / train_stats['std']
# normed_train_data = norm(train_dataset)
# normed_test_data = norm(test_dataset)
# 
# =============================================================================


x_scaler = MinMaxScaler()
train_dataset_scaled = x_scaler.fit_transform(train_dataset)
test_dataset_scaled = x_scaler.fit_transform(test_dataset)




def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

example_batch = train_dataset_scaled[:10]
example_result = model.predict(example_batch)
example_result

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  train_dataset_scaled,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])





