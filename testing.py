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
import csv


column_names = ['LOCATION', 'DAY', '0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
times = ['0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
rows = []
loc = ['WARRIGAL_RD N of HIGH STREET_RD', 'HIGH STREET_RD E of WARRIGAL_RD', 'WARRIGAL_RD S of HIGH STREET_RD', 'HIGH STREET_RD W of WARRIGAL_RD']
fields = ['LOCATION', 'DATE', 'TIME', 'VEHICLE COUNT']
#dict = ['LOCATION' : loc, 'DATE' :  ]
temp = []
dates=[]
datesA = ['01-10-06',
'02-10-06',
'03-10-06',
'04-10-06',
'05-10-06',
'06-10-06',
'07-10-06',
'08-10-06',
'09-10-06',
'10-10-06',
'11-10-06',
'12-10-06',
'13-10-06',
'14-10-06',
'15-10-06',
'16-10-06',
'17-10-06',
'18-10-06',
'19-10-06',
'20-10-06',
'21-10-06',
'22-10-06',
'23-10-06',
'24-10-06',
'25-10-06',
'26-10-06',
'27-10-06',
'28-10-06',
'29-10-06',
'30-10-06',
'31-10-06'
]
with open("./data/970_A.csv", 'r',encoding='utf-8') as data:
    reader = csv.reader(data)
    with open('./data/970_new1.csv', 'w', encoding='utf-8') as newfile:
        writer = csv.writer(newfile,delimiter=',')
       
        writer.writerow(fields)
        i=0
        j=0
        y=1
        for row in reader:
            
            rows.append(row)

              
   
        while (i < 4):
            while (j < 31):
                count = rows[y]
                del count[0:2]  
                for x in range(96):        
                    temp = [(loc[i]), datesA[j] ,(times[x]),(count[x])]
                    writer.writerow(temp)
                    j +=1
                    y+=1
             i += 1
        

        
    
dataset_path = "./data/970.csv"
raw_dataset = pd.read_csv(dataset_path)
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
my_data = pd.DataFrame(dataset)

# convert strings into np array
labels = np.array(map(float, column_names))
        
# Split data into training 80% and testing 20%
train_dataset = my_data.sample(frac = 0.8, random_state = 0)
test_dataset = my_data.drop(train_dataset.index)

        
# View some stats 
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
        
# Scale data
x_scaler = MinMaxScaler()
train_scaled = x_scaler.fit_transform(train_dataset)
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
predictions = Dense(1, activation='sigmoid')(x)
 
# # This creates a model that includes
# # the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mape'])

hist = model.fit( 
     train_scaled,
     labels, 
     batch_size=32, 
     epochs=100, verbose=2, 
     callbacks=None, 
     validation_split=0.05, 
     validation_data=None, 
     shuffle=True, 
     class_weight=None, 
     sample_weight=None, 
     initial_epoch=0)

predicted = model.predict(test_scaled)

model.save('Functional')
df = pd.DataFrame.from_dict(hist.history)
df.to_csv('./data/functional_loss.csv', encoding='utf-8', index=False)
    

# =============================================================================
# =============================================================================
# d = '2016-3-4 00:00'
# x = pd.date_range(d, periods=96, freq='15min')
# 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# 
# ax.plot(, y_true, label='True Data')
# for name, y_pred in zip(names, y_preds):
#     ax.plot(x, y_pred, label=name)
# 
#     plt.legend()
#     plt.grid(True)
#     plt.xlabel('Time of Day')
#     plt.ylabel('Flow')
# 
#     date_format = mpl.dates.DateFormatter("%H:%M")
#     ax.xaxis.set_major_formatter(date_format)
#     fig.autofmt_xdate()
# 
#     plt.show()
# =============================================================================
