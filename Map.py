#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:27:58 2019

@author: aseem
"""

import networkx as nx
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def DataFrame():
    # put coordinates in a DataFrame class
    df = pd.DataFrame(
        {'Point': ['970', '2000', '2200', '2820', '2825', '2827', '2846', 
                  '3001', '3002', '3120', '3122', '3126', '3127', '3180', 
                  '3662', '3682', '3685', '3804', '3812', '4030', '4032', 
                   '4034', '4035', '4040', '4043', '4051', '4057', '4063', 
                  '4262', '4263', '4264', '4266', '4270', '4272', '4273', 
                  '4321', '4324', '4335', '4812', '4821'],
         'Latitude': [-37.86703, -37.8516827, -37.8164799, -37.79477, -37.78661, -37.78093, -37.8612671, 
                     -37.81441, -37.81489, -37.82264, -37.82379, -37.82778, -37.82506, -37.79611, 
                     -37.80876, -37.83695, -37.85467, -37.83331, -37.83738, -37.79561, -37.80202, 
                     -37.81147, -37.8172654, -37.83256, -37.84683, -37.79419, -37.80431, -37.81404, 
                     -37.82155, -37.8228462, -37.82416, -37.82529, -37.82951, -37.83186, -37.84632, 
                     -37.800776, -37.809274, -37.80624, -37.82859, -37.81285],
         'Longitude': [145.09159, 145.0943457, 145.0977388, 145.03077, 145.06202, 145.07733, 145.058038, 
                      145.02243, 145.02663, 145.05734, 145.06466, 145.09885, 145.078, 145.08372, 
                       145.02757, 145.09699, 145.09384, 145.06247, 145.06119, 145.06251, 145.06127, 
                      145.05946, 145.0583603, 145.05545, 145.05275, 145.0696, 145.08197, 145.0801, 
                      145.01503, 145.0251292, 145.03445, 145.04387, 145.03304, 145.04668, 145.04378, 
                      145.0494611, 145.037306, 145.03518, 145.01644, 145.00849]
        })
    return df

def WithEdge(G):
    
    G.add_edge('970','2846')
    G.add_edge('2846','970')
    G.add_edge('970','3685')
    G.add_edge('3685','970')
    G.add_edge('3685','2000')
    G.add_edge('2000','3685')
    G.add_edge('2000','3682')
    G.add_edge('3682','2000')
    G.add_edge('2000','4043')
    G.add_edge('4043','2000')
    G.add_edge('3682','3126')
    G.add_edge('3126','3682')
    G.add_edge('3682','3804')
    G.add_edge('3804','3682') 
    G.add_edge('3126','3127')
    G.add_edge('3127','3126')
    G.add_edge('3127','4063')
    G.add_edge('4063','3127')
    G.add_edge('3127','3122')
    G.add_edge('3122','3127')
    G.add_edge('4043','4040')
    G.add_edge('4040','4043')
    G.add_edge('4043','4273')
    G.add_edge('4273','4043')
    G.add_edge('4040','3804')
    G.add_edge('3804','4040')
    G.add_edge('4040','3812')
    G.add_edge('3812','4040')
    G.add_edge('4040','4272')
    G.add_edge('4272','4040')
    G.add_edge('4040','3120')
    G.add_edge('3120','4040')
    G.add_edge('3804','3812')
    G.add_edge('3812','3804')
    G.add_edge('3804','3122')
    G.add_edge('3122','3804')
    G.add_edge('3122','3120')
    G.add_edge('3120','3122')
    G.add_edge('4272','4273')
    G.add_edge('4273','4272')
    G.add_edge('4272','4270')
    G.add_edge('4270','4272')
    G.add_edge('4270','4812')
    G.add_edge('4812','4270')
    G.add_edge('4270','4264')
    G.add_edge('4264','4270')
    G.add_edge('4264','4263')
    G.add_edge('4263','4264')
    G.add_edge('4264','4266')
    G.add_edge('4266','4264')
    G.add_edge('4264','4324')
    G.add_edge('4324','4264')
    G.add_edge('4324','4034')
    G.add_edge('4034','4324')
    G.add_edge('3120','4035')
    G.add_edge('4035','3120')
    G.add_edge('4035','4034')
    G.add_edge('4034','4035')
    G.add_edge('4035','3002')
    G.add_edge('3002','4035')
    G.add_edge('4034','4032')
    G.add_edge('4032','4034')
    G.add_edge('4324','3662')
    G.add_edge('3662','4324') 
    G.add_edge('3662','3001')
    G.add_edge('3001','3662') 
    G.add_edge('3662','4335')
    G.add_edge('4335','3662') 
    G.add_edge('3662','2820')
    G.add_edge('2820','3662')
    G.add_edge('3001','4262')
    G.add_edge('4262','3001')
    G.add_edge('3001','4821')
    G.add_edge('4821','3001')
    G.add_edge('4321','2820')
    G.add_edge('2820','4321')
    G.add_edge('4321','4335')
    G.add_edge('4335','4321')
    G.add_edge('4321','4032')
    G.add_edge('4032','4321')
    G.add_edge('4321','4030')
    G.add_edge('4030','4321')
    G.add_edge('4032','4030')
    G.add_edge('4030','4032')
    G.add_edge('4032','4057')
    G.add_edge('4057','4032')
    G.add_edge('4030','2825')
    G.add_edge('2825','4030')
    G.add_edge('4030','4051')
    G.add_edge('4051','4030')
    G.add_edge('4051','2827')
    G.add_edge('2827','4051')
    G.add_edge('4051','3180')
    G.add_edge('3180','4051')
    G.add_edge('4057','4063')
    G.add_edge('4063','4057')
    G.add_edge('4057','3180')
    G.add_edge('3180','4057')
    G.add_edge('4063','2200')
    G.add_edge('2200','4063')
    G.add_edge('4063','4034')
    G.add_edge('4034','4063')
    G.add_edge('2825','2827')
    G.add_edge('2827','2825')
    G.add_edge('3002','3001')
    G.add_edge('3001','3002')
    G.add_edge('3002','3662')
    G.add_edge('3662','3002')
    G.add_edge('3002','4263')
    G.add_edge('4263','3002')
    
    return G

def QuickestPath(G, source, target, df):
    sp = nx.shortest_path(G, source=source, target=target)

    distTotal= 0
    for i in range(len(sp)-1):
        l1 = df[df['Point']==sp[i]].index.values.astype(int)
        l2 = df[df['Point']==sp[i+1]].index.values.astype(int)
        distIndividual = (geodesic((df.iloc[l1[0]].Latitude, df.iloc[l1[0]].Longitude), (df.iloc[l2[0]].Latitude, df.iloc[l2[0]].Longitude)).kilometers)
        distTotal += distIndividual
        
    return ((len(sp)), sp, round(distTotal, 2))

def AlternatePaths(G, source, target, df, n):
    alsp = nx.all_simple_paths(G, source=source, target=target)
    path = (list(alsp)[n])
    
    distTotal= 0
    for i in range(len(path)-1):
        l1 = df[df['Point']==path[i]].index.values.astype(int)
        l2 = df[df['Point']==path[i+1]].index.values.astype(int)
        distIndividual = (geodesic((df.iloc[l1[0]].Latitude, df.iloc[l1[0]].Longitude), (df.iloc[l2[0]].Latitude, df.iloc[l2[0]].Longitude)).kilometers)
        distTotal += distIndividual
    return ((len(path)), path, round(distTotal, 2))
    
def main():
    source = '2827'
    target = '4063'
    
    dataFrame = DataFrame()
    
    G=nx.Graph()
    for index, row in dataFrame.iterrows():
        G.add_node(row.Point, y = row.Latitude , x = row.Longitude, 
                   geometry=(row.Longitude, row.Latitude))
        
    G = WithEdge(G)
    intersections, path, distance = QuickestPath(G, source, target, dataFrame)
    
    print("SHORTEST PATH is via these " + str(intersections) + 
          " intersection points \ntotal distance travelled " 
          + str(distance) + " KM\n" + str(path) + "\n")
    
    print("ALTERNATIVE PATHs are\n")
    alsp = nx.all_simple_paths(G, source=source, target=target)
    number_of_paths = min(4,(len((list(alsp)))))
    for i in range(number_of_paths):
        intersections, path, distance = AlternatePaths(G, source, target, dataFrame, i)
        print("Via these " + str(intersections) + 
          " intersection points \ntotal distance travelled " 
          + str(distance) + " KM\n" + str(path) + "\n")
            
if __name__ == '__main__':
    main()