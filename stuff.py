# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:21:52 2019

@author: User
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

file = './data/simple.csv'

file1 = pd.read_csv(file)

train, test = train_test_split(file1, test_size=0.3, shuffle=False)
test.to_csv('./data/test_basic.csv', encoding='utf-8', index=False)
train.to_csv('./data/train_basic.csv', encoding='utf-8', index=False)
 