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
        #self.fullPrediction()
    
    # Calculate travel time based on distance + traffic flow & display in GUI
    #def calculateTravelTime(self, count, distance):
        
    
    def fullPrediction(self):
        st = np.int64(self.startTime.text)
        et = np.int64(self.endTime.text)
        my_day = np.int64(self.day.text)
        d = self.direction.text
#        start = np.int64(self.startScats.text)
#        end = np.int64(self.endScats.text)
        predictionClass = CleanPrediction()

        pathData = Map.generatePaths(self.startScats.text, self.endScats.text)
        shortest_route = pathData[0]
        shortestMap = pathData[1]
        allPaths = pathData[2]
        allDistances = pathData[3]
        self.ids.mapFigure.remove_widget(FigureCanvasKivyAgg(self.plt.gcf()))
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(shortestMap.gcf()))
        currentPrediction = 0
        newPrediction = 0
        lowestTrafficPath = 0,0
        currentPath = 0
        for path in allPaths:
            currentPath += 1
            print("Current Loop: " + str(currentPath))
            for scats in path:
                print("Current Scats: " +  scats)
                newPrediction, metrics = predictionClass.predict(np.int64(scats), st, et, my_day, d)
#               
                mape = metrics[0]
                print(mape)
                print(newPrediction)
                calculateTravelTime(newPrediction, scats, path[scats -1]) # Get distance between 2 SCATS?
            if currentPath == 1:
                currentPrediction = newPrediction
            else:
                if int(newPrediction) < int(currentPrediction):
                    currentPrediction = newPrediction
                    # Can you use this to calculate travel time?
                    lowestTrafficPath = currentPath,currentPrediction
            print("Current Lowest Trafic: " + str(lowestTrafficPath))

        self.output.text = str(pathData[0])


class MapFigure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super(MapFigure, self).__init__(plt.gcf(), **kwargs)

class testApp(App):
    def build(self):
        
        return Menu()


if __name__ == "__main__":
    testApp().run()