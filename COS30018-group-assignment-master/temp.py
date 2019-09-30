# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn import tree
import numpy as np
import pandas as pd

data=pd.read_csv('970.csv')

sun=data.loc[data['day'] == 0]
mon=data.loc[data['day'] == 1]
tue=data.loc[data['day'] == 2]
wed=data.loc[data['day'] == 3]
thur=data.loc[data['day'] == 4]
fri=data.loc[data['day'] == 5]
sat=data.loc[data['day'] == 6]

features=[
        [sun.iloc[0,1],sun.iloc[0,0],0],
        [sun.iloc[0,1],sun.iloc[0,0],0],
        [sun.iloc[0,1],sun.iloc[0,0],0],
        [sun.iloc[0,1],sun.iloc[0,0],0],
        [sun.iloc[0,1],sun.iloc[0,0],0],
        [mon.iloc[0,1],mon.iloc[0,0],0],
        [mon.iloc[0,1],mon.iloc[0,0],0],
        [mon.iloc[0,1],mon.iloc[0,0],0],
        [mon.iloc[0,1],mon.iloc[0,0],0],
        [mon.iloc[0,1],mon.iloc[0,0],0]
        ]
lables=[
        sun.iloc[0,2],
        sun.iloc[1,2],
        sun.iloc[2,2],
        sun.iloc[3,2],
        sun.iloc[4,2],
        mon.iloc[0,3],
        mon.iloc[1,3],
        mon.iloc[2,3],
        mon.iloc[3,3],
        mon.iloc[4,3]
        ]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,lables)
print(clf.predict([[1,970,0]]))
