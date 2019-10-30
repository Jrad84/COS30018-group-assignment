"""
@author: Daniel Shawyer
"""

import kivy 
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch')
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
import networkx as nx

import contextily as ctx 
import numpy as np
from fiona.crs import from_epsg
from kivy.garden.matplotlib import FigureCanvasKivy, FigureCanvasKivyAgg
import matplotlib.pyplot as plt
# from kivy.config import Config
from kivy.core.window import Window

# from dfguik import DfguiWidget
import Map
import DrawMap
from CleanPrediction import CleanPrediction


class Menu(Widget):
   
    def __init__(self, **kwargs):
        super(Menu, self).__init__(**kwargs)
        Window.size = (1024, 800)
        data = DrawMap.DataFrame()
        self.route = []
        
        #print(self.route[0])
        self.plt = Map.CreateInitialMap(data)
        self.run()
        
    startScats = ObjectProperty(None)
    endScats = ObjectProperty(None)
    day = ObjectProperty(None)
    startTime = ObjectProperty(None)
    endTime = ObjectProperty(None)
    direction = ObjectProperty
    t1 = ObjectProperty(None)
    t2 = ObjectProperty(None)
    t3 = ObjectProperty(None)
    t4 = ObjectProperty(None)
    t5 = ObjectProperty(None)
    
    
    bestRoute = ObjectProperty(None)

    
    def run(self):
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(self.plt.gcf()))
        

  

    # Calculate travel time based on distance + traffic flow & display in GUI
    def calculateTravelTime(self, counts, path, distance):
        currentTime = 1
        size = len(path)
        i = 0
        #for i in len(path):
        while (i < size):    
            # Add 10 seconds per car
            currentTime += counts[str(path[i])] + 10 
            # Add 30 sec to pass through each intersection
            currentTime += 30
            i += 1
                  
        currentTime += distance * 60 #km/hr

        return currentTime
        
    def fullPrediction(self):
       
        st = np.int64(self.startTime.text)
        et = np.int64(self.endTime.text)
        my_day = np.int64(self.day.text)
        d = self.direction.text
        if self.t1.active:
            name = 'GRU'
        if self.t2.active:
            name = 'LSTM'
        if self.t3.active:
            name = 'SAES'
        if self.t4.active:
            name='RNN'
        if self.t5.active:
            name='BI'
        predictionClass = CleanPrediction()
        counts = {}
        mape={}
        finalPathMape={}
        pathData = Map.generatePaths(self.startScats.text, self.endScats.text)      
        shortestMap = pathData[1]
        allPaths = pathData[2]
        allDistances = np.int64(pathData[3])
        
        self.ids.mapFigure.remove_widget(FigureCanvasKivyAgg(self.plt.gcf()))
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(shortestMap.gcf()))

        newPrediction = 0

        firstRun = True
        i=0
        j=0

        # For each path in possible paths
        for path in allPaths:
            # Get prediction for each intersection
            for scats in path:
                newPrediction = predictionClass.predict(int(scats), st, et, my_day, d,name)
                mape[str(scats)]=predictionClass.metrics[0]
                counts[str(scats)] = newPrediction
                j +=1           
            j=0
            distance = allDistances[i]
            # Calculate travel time based on distance + traffic
            currentTime = int(self.calculateTravelTime(counts, path, distance))

            if firstRun:
                bestTime = currentTime
                bestPath = path
                firstRun = False
            elif currentTime < bestTime:
                bestTime = currentTime
                finalPathMape=mape
                bestPath = path
            i += 1
        bestTime = round((bestTime / 60), 2)
        self.bestRoute.text = "Via intersections: " + str(bestPath)+"\n\n" +"Travel time: "+ str(bestTime) + " minutes"+"\n\n MAPE for each Intersection: "+str(finalPathMape)+"\n\n Average MAPE: "+str(sum(finalPathMape.values())/len(finalPathMape))

class MapFigure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super(MapFigure, self).__init__(plt.gcf(), **kwargs)

class testApp(App):
    def build(self):
        
        return Menu()


if __name__ == "__main__":
    testApp().run()