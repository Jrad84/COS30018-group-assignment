# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:15:38 2019

@author: JT
"""

import csv
import re

fields = ['SCATS', 'LOCATION', 'DATE', 'TIME', 'COUNT']
rows = []
times = []
dates = []
dir = ['N', 'E', 'S', 'W']
direction = ['N{1}', 'E{1}', 'S{1}', 'W{1}']
loc = []
count = []
scats = []
periods = 96
num_rows = 4192
temp = []

with open("scats-data.csv", 'r',encoding='utf-8',newline='') as data:
    reader = csv.reader(data)
    with open('V2.csv', 'w', encoding='utf-8',newline='') as newfile:
        writer = csv.writer(newfile,delimiter=',')
       
        writer.writerow(fields) # write column names
        
        for row in reader:      
            rows.append(row)
        
        j = 0
        y = 1
        z=0
        while (z < 4):
            while (j < num_rows):
                i = 0
            
                count = rows[y].copy()
                del count[0:3]
                loc = rows[y].copy()
                del loc[2:]
                times = rows[0].copy()
                del times[0:3]
                scats = rows[y].copy()
                del scats[1:]
                dates = rows[y].copy()
                del dates[3:]
                del dates[0:2]
                #dates[0].replace('-','')
                while (i < periods):
                    temp = [(scats[0]), (loc[1]), (dates[0].replace('-','')), (times[i].replace(':','')), count[i]]
                    writer.writerow(temp)
                    i+=1
                y+=1
                j+=1
            z+=1
         
                
            
        