# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:58:09 2019

@author: User
"""
import re
import pandas as pd
from datetime import datetime

file1 = "simple.csv"
file = pd.read_csv(file1)
regex = '\d{2}-\d{2}-\d{4}'

time = []
date = []
times = file['DATE_TIME'].copy()
dates = file['DATE_TIME'].copy()
for row in times:
        x = datetime.strptime(row, '%d-%m-%y %H:%M').date()
       
        format = '%H%M'
        my_format = '%d-%m-%y'
        x.strftime(my_format)
        date.append(x)
        line = re.sub('\\d{2}-\\d{2}-\\d{2}', '', row)
        #y = line.replace(':', '')
        time.append(line)

file.insert(2, 'TIME', time, True)
file.insert(1, 'DATE', date, True)

file.to_csv('simple_data.csv', encoding='utf-8', index=False)