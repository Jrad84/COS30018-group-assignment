import kivy 
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

class Menu(Widget):
    scats = ObjectProperty(None)
    day = ObjectProperty(None)
    startTime = ObjectProperty(None)
    endTime = ObjectProperty(None)
    direction = ObjectProperty(None)
    prediction = ObjectProperty(None)
    
    def btn(self):
        print("Scats: ", self.scats.text, "Day: ", self.day.text, "Start Time: ", self.startTime.text, "End time: ", self.endTime.text, "Direction: ", self.direction.text, "Prediction: ", self.prediction.text)


class testApp(App):
    def build(self):
        return Menu()

        
if __name__ == "__main__":
    testApp().run()