# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:48:57 2019

@author: User
"""

import csv

rows = []
loc = []
num_rows = 402433
dir = ['N', 'E', 'S', 'W']
freq = 2975

with open("V1.csv", 'r',encoding='utf-8',newline='') as data:
    reader = csv.reader(data)
    with open('V2.csv', 'w', encoding='utf-8',newline='') as newfile:
        writer = csv.writer(newfile,delimiter=',')
        
        for row in reader:    
            del row[2:]
            rows.append(row)
        for x in range(33):
            i = 0    
            for row in rows:
            
                while (i < 4):
                    j = 0
                    while (j < freq):
                        writer.writerow(dir[i])
                        j+=1
                    i+=1

            
            
        
    
    
      
        
            