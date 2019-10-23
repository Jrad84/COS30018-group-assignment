#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:27:58 2019

@author: Aseem with modifications by Daniel Shawyer
"""
import DrawMap
import networkx as nx
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, LineString
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from fiona.crs import from_epsg
import contextily as ctx 
from itertools import islice

# converts 4326 to 3857
def CRSConverter(geoData):
    geoData.crs = from_epsg(4326)
    returnData = geoData.to_crs(3857)
    return returnData

# add distance between each point in km using geodesic
def DistanceTravelled(path, df):
    distTotal= 0
    for i in range(len(path)-1):
        l1 = df[df['Point']==path[i]].index.values.astype(int)
        l2 = df[df['Point']==path[i+1]].index.values.astype(int)
        distIndividual = (geodesic((df.iloc[l1[0]].Latitude, df.iloc[l1[0]].Longitude), (df.iloc[l2[0]].Latitude, df.iloc[l2[0]].Longitude)).kilometers)
        distTotal += distIndividual
    return distTotal
    
# Prints the whole map with 40 intersections
def InitialMap(df):
    
    # convert data frane to geo data frame
    gdf = geopandas.GeoDataFrame(
     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
    
    # changes epsg for plotting
    gdf2 = CRSConverter(gdf)
    
    # plot map, add text as Point and basemap from network x
    ax = gdf2.plot(figsize=(10, 10), alpha=0.5, edgecolor='blue')
    for tuples in gdf2.itertuples():
        plt.text(tuples.geometry.x, tuples.geometry.y, tuples.Point)
            
    ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite, zoom=12)
    ax.set_axis_off()
    plt.show()

def CreateInitialMap(df):
    
    # convert data frane to geo data frame
    gdf = geopandas.GeoDataFrame(
     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
    
    # changes epsg for plotting
    gdf.crs = from_epsg(3857)
    
    # plot map, add text as Point and basemap from network x
    ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='blue')
    for tuples in gdf.itertuples():
        plt.text(tuples.geometry.x, tuples.geometry.y, tuples.Point)
            
    # ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite, zoom=12)
    ax.set_axis_off()
    # plt.show()
    return plt


    
# prints the shortest path map    
def PathMap(sp, df):
    # stores lines/roads/edges of sp (shortest path) 
    listLine = []
    
    for i in range(len(sp)-1):
        # l1 and l2 are row number of each point in data frame
        l1 = df[df['Point']==sp[i]].index.values.astype(int)
        l2 = df[df['Point']==sp[i+1]].index.values.astype(int)
        # from row number we get geometry and form list of line
        # listLine.append(LineString([df.iloc[l1[0]].geometry, df.iloc[l2[0]].geometry]))
        point1 = df.iloc[l1[0]].Latitude,df.iloc[l1[0]].Longitude
        point2 = df.iloc[l2[0]].Latitude,df.iloc[l2[0]].Longitude
        listLine.append(LineString([point1, point2]))
    
    # convert line to dataframe then to geo df
    dfLine = pd.DataFrame({'geometry': listLine})   
    gdfLine = geopandas.GeoDataFrame(dfLine)
    
    # get orignal df and change to crs
    # this will be helpful to mark intersection on path map that only has line
    gdf = geopandas.GeoDataFrame(
     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

    # gdf2 = CRSConverter(gdf)
    # gdfLine2 = CRSConverter(gdfLine)

    gdf.crs = from_epsg(3857)
    gdfLine.crs = from_epsg(3857)
    
    ax = gdfLine.plot(figsize=(10, 10), alpha=1, edgecolor='red')

    for a in gdf.itertuples():
          if sp.__contains__(a.Point):
              plt.text(a.geometry.x, a.geometry.y, a.Point)
              
    # ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite, zoom=12)
    ax.set_axis_off()
    plt.show()

# prints the shortest path map    
def CreateMapWithPaths(sp, df):
    # stores lines/roads/edges of sp (shortest path) 
    listLine = []
    
    for i in range(len(sp)-1):
        # l1 and l2 are row number of each point in data frame
        l1 = df[df['Point']==sp[i]].index.values.astype(int)
        l2 = df[df['Point']==sp[i+1]].index.values.astype(int)
        # from row number we get geometry and form list of line
        point1 = df.iloc[l1[0]].Latitude,df.iloc[l1[0]].Longitude
        point2 = df.iloc[l2[0]].Latitude,df.iloc[l2[0]].Longitude
        listLine.append(LineString([point1, point2]))
    
    # convert line to dataframe then to geo df
    dfLine = pd.DataFrame({'geometry': listLine})   
    gdfLine = geopandas.GeoDataFrame(dfLine)
    
    # get orignal df and change to crs
    # this will be helpful to mark intersection on path map that only has line
    gdf = geopandas.GeoDataFrame(
     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

    gdf.crs = from_epsg(3857)
    gdfLine.crs = from_epsg(3857)
    
    ax = gdfLine.plot(figsize=(10, 10), alpha=1, edgecolor='red')

    for a in gdf.itertuples():
          if sp.__contains__(a.Point):
              plt.text(a.geometry.x, a.geometry.y, a.Point)
              
    # ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite, zoom=12)
    ax.set_axis_off()
    # plt.show()
    return plt
    

def QuickestPath(G, source, target, df):
    # network x finds shortest path as list of nodes/points/intersections
    sp = nx.shortest_path(G, source=source, target=target)

    # add distance between each point in km using geodesic
    distTotal= DistanceTravelled(sp, df)        
    # PathMap(sp, df)    
    return ((len(sp)), sp, round(distTotal, 2))



# n in the loop number (0, 1, 2, 3)
# spl shortest path length
def AlternatePaths(G, source, target, df, n, spl):
    # finds a path
    
    alsp = nx.all_simple_paths(G, source=source, target=target, cutoff=spl+5)
    path = (list(alsp)[n])
    
    distTotal= DistanceTravelled(path, df)
    # clculates distance and returns the path
    PathMap(path, df)  
    return ((len(path)), path, round(distTotal, 2))
    
def main():
    source = '970'
    target = '4032'
    
    dataFrame = DrawMap.DataFrame()
    print(dataFrame)
    InitialMap(dataFrame)
    
    # creats an empty graph then adds rows (point, lat, long, geometry)
    G=nx.Graph()
    for index, row in dataFrame.iterrows():
        G.add_node(row.Point, y = row.Latitude , x = row.Longitude, 
                   geometry=(row.Longitude, row.Latitude))
    
    # adds edges roads   
    G = DrawMap.WithEdge(G)
    intersections, path, distance = QuickestPath(G, source, target, dataFrame)
    # shortest path length
    spl = len(path)
    
    print("SHORTEST PATH is via these " + str(intersections) + 
          " intersection points \ntotal distance travelled " 
          + str(distance) + " KM\n" + str(path) + "\n")
    
    # genrates all possible paths (1000s) as a gentaror type
    alsp = nx.all_simple_paths(G, source=source, target=target)
    # max 4 more paths (could be any 4, random)
    number_of_paths = min(4,(len((list(alsp)))))
    #prints upto max 4 paths
    for i in range(number_of_paths):
        intersections, path, distance = AlternatePaths(G, source, target, dataFrame, i, spl)
        print("Alternative path via these " + str(intersections) + 
          " intersection points \ntotal distance travelled " 
          + str(distance) + " KM\n" + str(path) + "\n")


def createRoute(start, end):
    source = start
    target = end
    # dataFrame = dataframe
    
    dataFrame = DrawMap.DataFrame()
    # CreateInitialMap(dataFrame)
    
    # creats an empty graph then adds rows (point, lat, long, geometry)
    G=nx.Graph()
    for index, row in dataFrame.iterrows():
        G.add_node(row.Point, y = row.Latitude , x = row.Longitude, 
                   geometry=(row.Longitude, row.Latitude))
    
    # adds edges roads   
    G = DrawMap.WithEdge(G)
    intersections, path, distance = QuickestPath(G, source, target, dataFrame)
    # shortest path length
    
    print("SHORTEST PATH is via these " + str(intersections) + 
          " intersection points \ntotal distance travelled " 
          + str(distance) + " KM\n" + str(path) + "\n")
    return path, CreateMapWithPaths(path,dataFrame)

def generatePaths(start, end):
    dataFrame = DrawMap.DataFrame()

    G=nx.Graph()
    for index, row in dataFrame.iterrows():
        G.add_node(row.Point, y = row.Latitude , x = row.Longitude, 
                   geometry=(row.Longitude, row.Latitude))
    
    # adds edges roads   
    G = DrawMap.WithEdge(G)
    intersections, shortestPath, distance = QuickestPath(G, start, end, dataFrame)
    # shortest path length
    
    print("SHORTEST PATH is via these " + str(intersections) + 
          " intersection points \ntotal distance travelled " 
          + str(distance) + " KM\n" + str(shortestPath) + "\n")

    allShortestPaths = k_shortest_paths(G, start, end, 5)

      
    print([p for p in allShortestPaths])
    allDistances=[]
    for path in allShortestPaths:  
        allDistances.append(DistanceTravelled(path, dataFrame))
    # calculates distance and returns the path 
    print([p for p in allDistances])

    #cardinality list


    return shortestPath, CreateMapWithPaths(shortestPath,dataFrame), allShortestPaths, allDistances

# Based on algorithm by Jin Y. Yen Finding the first K paths requires O(KN3) operations. Ref: “Finding the K Shortest Loopless Paths in a Network”, Management Science, Vol. 17, No. 11, Theory Series (Jul., 1971), pp. 712-716.
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def cardinality(point1, point2):
      
    df = DrawMap.DataFrame()
    point1Index = df[df['Point']==point1].index.values.astype(int)
    point2Index = df[df['Point']==point2].index.values.astype(int)
    lat2 = df.iloc[point2Index[0]].Latitude
    lat1 = df.iloc[point1Index[0]].Latitude
    
    latDiff = lat2 - lat1
    if (latDiff > 0):
        NS = 'N'
    else:
        NS = 'S'
    
    long2 = df.iloc[point2Index[0]].Longitude
    long1 = df.iloc[point1Index[0]].Longitude
    
    longDiff = long2 - long1
    if (longDiff > 0):
        EW = 'E'
    else:
        EW = 'W'
    
    if ((abs(latDiff) - abs(longDiff)) > 0.0):
        return NS
    else:
        return EW