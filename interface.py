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
import geopandas
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
        
        # print(self.route[0])
        self.plt = Map.CreateInitialMap(data)
        self.run()
        
    startScats = ObjectProperty(None)
    endScats = ObjectProperty(None)
    day = ObjectProperty(None)
    startTime = ObjectProperty(None)
    endTime = ObjectProperty(None)
    direction = ObjectProperty(None)
    output = ObjectProperty(None)
    # model = ObjectProperty(None)
    
    def run(self):
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(self.plt.gcf()))

    
    def predictButton(self):
        st = np.int64(self.startTime.text)
        et = np.int64(self.endTime.text)
        my_day = np.int64(self.day.text)
        d = self.direction.text
        self.output.text = ""
        for scats in self.route[0]:
            scats = np.int64(scats)
            print(scats)
            predictionClass = CleanPrediction()
            prediction = predictionClass.predict(scats, st, et, my_day, d)
            self.output.text += str(prediction) + " "

    def routeButton(self):
        self.route = Map.createRoute(self.startScats.text, self.endScats.text)
        self.output.text = str(self.route[0])
    
    def fullPredction(self):
        pathData = Map.generatePaths(self.startScats.text, self.endScats.text)
        shortest_route = pathData[0]
        shortestMap = pathData[1]
        allPaths = pathData[2]
        allDistances = pathData[3]
        self.ids.mapFigure.remove_widget(FigureCanvasKivyAgg(self.plt.gcf()))
        self.ids.mapFigure.add_widget(FigureCanvasKivyAgg(shortestMap.gcf()))
        
        self.output.text = str(pathData[0])


class MapFigure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super(MapFigure, self).__init__(plt.gcf(), **kwargs)

class testApp(App):
    def build(self):
        
        return Menu()


if __name__ == "__main__":
    testApp().run()