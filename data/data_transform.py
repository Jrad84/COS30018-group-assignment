
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:36:22 2019

@author: JT
"""
import csv

times = ['0:00', '0:15', '0:30', '0:45', '1:00', '1:15', '1:30', '1:45', '2:00', '2:15', '2:30', '2:45', '3:00', '3:15', '3:30', '3:45', '4:00', '4:15', '4:30', '4:45', '5:00', '5:15', '5:30', '5:45', '6:00', '6:15', '6:30', '6:45', '7:00', '7:15',	'7:30',	'7:45',	'8:00',	'8:15',	'8:30',	'8:45',	'9:00',	'9:15',	'9:30',	'9:45',	'10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30', '11:45', '12:00', '12:15', '12:30', '12:45', '13:00', '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00', '15:15', '15:30', '15:45', '16:00', '16:15', '16:30', '16:45', '17:00', '17:15', '17:30', '17:45', '18:00', '18:15', '18:30', '18:45', '19:00', '19:15', '19:30', '19:45', '20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15', '23:30', '23:45' ]
rows = []
#times = []
loc = ['WARRIGAL_RD N of HIGH STREET_RD', 'HIGH STREET_RD E of WARRIGAL_RD', 'WARRIGAL_RD S of HIGH STREET_RD', 'HIGH STREET_RD W of WARRIGAL_RD']
fields = ['LOCATION', 'DATE', 'TIME', 'VEHICLE COUNT']
#dict = ['LOCATION' : loc, 'DATE' :  ]
temp = []
dates=[]
num_rows = 123 # number of rows in csv file

with open("./data/970_A.csv", 'r',encoding='utf-8') as data:
    reader = csv.reader(data)
    with open('./data/970_new1.csv', 'w', encoding='utf-8') as newfile:
        writer = csv.writer(newfile,delimiter=',')
       
        writer.writerow(fields)
        i=0
        j=0
        y=0

        for row in reader:      
            rows.append(row)
            #del rows[0]
        for row in rows:
            while (i < 4):  # 4 locations 
                while (j < 31):   # each day of month ()
                    #loc = rows[j].copy()
                    while (y < num_rows): 
                        dates = rows[j].copy()
                        del dates[2:98]
                        del dates[0]
                        count = rows[y].copy() 
                        del count[0:2]
                        for x in range(96):  # each time interval
                            temp = [(loc[i]), (dates[0]) ,(times[x]),(count[x])]
                            writer.writerow(temp)
                        y+=1
                        j+=1
                        
                i+=1