"""
@author: Daniel Shawyer
"""

import kivy 
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
    direction = ObjectProperty(None)
    output = ObjectProperty(None)
    getCount = ObjectProperty(None)
    mape = ObjectProperty(None)
    bestRoute = ObjectProperty(None)
    time = ObjectProperty(None)

    
    def run(self):
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(self.plt.gcf()))

    def predictButton(self):
        st = np.int64(self.startTime.text)
        et = np.int64(self.endTime.text)
        my_day = np.int64(self.day.text)
        d = self.direction.text
        self.getCount.text = ""
        predictionClass = CleanPrediction()
        scats = np.int64(self.startScats.text)

        prediction, metrics = predictionClass.predict(scats, st, et, my_day, d)
        self.getCount.text += str(prediction) + " "
        self.mape.text += str(metrics[0]) + "%"
#        for scats in self.route[0]:
#            scats = np.int64(scats)
#            print(scats)
#            prediction = predictionClass.predict(scats, st, et, my_day, d)
#            self.output.text += str(prediction) + " "

    def routeButton(self):
        self.route = Map.createRoute(self.startScats.text, self.endScats.text)
        self.output.text = str(self.route[0])
    
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

        predictionClass = CleanPrediction()
        counts = {}
        pathData = Map.generatePaths(self.startScats.text, self.endScats.text)      
        shortestMap = pathData[1]
        allPaths = pathData[2]
        allDistances = np.int64(pathData[3])
        
        self.ids.mapFigure.remove_widget(FigureCanvasKivyAgg(self.plt.gcf()))
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(shortestMap.gcf()))

        newPrediction = 0

        times = []
        firstRun = True
        i=0
        for path in allPaths:

            for scats in path:

                newPrediction = predictionClass.predict(np.int64(scats), st, et, my_day, d)
                counts[str(scats)] = newPrediction
                
            distance = allDistances[i]    
            currentTime = int(self.calculateTravelTime(counts, path, distance))
            if firstRun:
                bestTime = currentTime
                bestPath = path
                firstRun = False
            elif currentTime < bestTime:
                bestTime = currentTime
                bestPath = path
            i += 1
        bestTime = round((bestTime / 60), 2)
        self.bestRoute.text = "Via intersections: " + str(bestPath)
        self.time.text = "Travel time: " + str(bestTime) + " minutes"

class MapFigure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super(MapFigure, self).__init__(plt.gcf(), **kwargs)

class testApp(App):
    def build(self):
        
        return Menu()


if __name__ == "__main__":
    testApp().run()