#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:27:58 2019

@author: aseem
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
    
# prints the shortest path map    
def PathMap(sp, df):
    # stores lines/roads/edges of sp (shortest path) 
    listLine = []
    
    for i in range(len(sp)-1):
        # l1 and l2 are row number of each point in data frame
        l1 = df[df['Point']==sp[i]].index.values.astype(int)
        l2 = df[df['Point']==sp[i+1]].index.values.astype(int)
        # from row number we get geometry and form list of line
        listLine.append(LineString([df.iloc[l1[0]].geometry, df.iloc[l2[0]].geometry]))
    
    # convert line to dataframe then to geo df
    dfLine = pd.DataFrame({'geometry': listLine})   
    gdfLine = geopandas.GeoDataFrame(dfLine)
    
    # get orignal df and change to crs
    # this will be helpful to mark intersection on path map that only has line
    gdf = geopandas.GeoDataFrame(
     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

    gdf2 = CRSConverter(gdf)
    gdfLine2 = CRSConverter(gdfLine)
    
    ax = gdfLine2.plot(figsize=(10, 10), alpha=1, edgecolor='red')

    for a in gdf2.itertuples():
          if sp.__contains__(a.Point):
              plt.text(a.geometry.x, a.geometry.y, a.Point)
              
    ctx.add_basemap(ax, url=ctx.providers.Stamen.TonerLite, zoom=12)
    ax.set_axis_off()
    plt.show()

def QuickestPath(G, source, target, df):
    # network x finds shortest path as list of nodes/points/intersections
    sp = nx.shortest_path(G, source=source, target=target)

    # add distance between each point in km using geodesic
    distTotal= DistanceTravelled(sp, df)        
    PathMap(sp, df)    
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
            
if __name__ == '__main__':
    main()