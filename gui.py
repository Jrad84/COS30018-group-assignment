# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:20:30 2019

@author: User
"""

from tkinter import *
import tkinter as tk



root = Tk()

def predict():
    scats = e1.get()
    dir = e2.get()
    day = e3.get()
    
    
    e1.delete(0, tk.END)
    e2.delete(0, tk.END)

master = tk.Tk()
tk.Label(master, text="Enter Scats").grid(row=0)
tk.Label(master, text="Enter direction eg N, E, S, W").grid(row=1)
tk.Label(master, text="Enter day eg mon, tue").grid(row=2)

e1 = tk.Entry(master)
e2 = tk.Entry(master)
e3 = tk.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)

tk.Button(master, 
          text='Quit', 
          command=master.quit).grid(row=4, 
                                    column=0, 
                                    sticky=tk.W, 
                                    pady=4)

tk.Button(master, text='Show', command=predict).grid(row=4, 
                                                               column=1, 
                                                               sticky=tk.W, 
                                                               pady=4)

master.mainloop()

tk.mainloop()



#Lb1 = Listbox(root, selectmode=SINGLE)
#for i in range (38):
#    Lb1.insert(i + 1, scats[i])
#
#Lb2 = Listbox(root, selectmode=SINGLE)
#  
#for i in range (96):
#    Lb2.insert(i + 1, times[i])
#    
#Lb3= Listbox(root, selectmode=SINGLE)
#    
#for i in range (96):
#    Lb3.insert(i + 1, times[i])
#    
#Lb4 = Listbox(root, selectmode=SINGLE)
#    
#for i in range (7):
#    Lb4.insert(i + 1, days[i])
#        
#Lb5 = Listbox(root)
#Lb5.insert(1, 'N')
#Lb5.insert(2, 'S')
#Lb5.insert(3, 'E')
#Lb5.insert(4, 'W')
#    
#Lb6 = Listbox(root)
#Lb6.insert(1, 'LSTM')
#Lb6.insert(2, 'GRU')
#Lb6.insert(3, 'SAES')
#    
#label1 = Label(root, text='SCATS')
#label1.grid(row=0, column=0)
#    
#label2 = Label(root, text='Start')
#label2.grid(row=1, column=0)
#    
#label3 = Label(root, text='End')
#label3.grid(row=1, column=1)
#    
#label4 = Label(root, text='Day')
#label4.grid(row=0, column=1)
#    
#label5 = Label(root, text='Direction')
#label5.grid(row=2, column=0)
#    
#label6 = Label(root, text='Model')
#label6.grid(row=2, column=1)
#    
#b = Button(root, text='Go')
#b.grid(row=3,column=0)
#    
#Lb1.grid(row=0, column=0)
#Lb2.grid(row=1, column=0)
#Lb3.grid(row=1, column=1)
#Lb4.grid(row=0, column=1)
#Lb5.grid(row=2, column=0)
#Lb6.grid(row=2, column=1)
#    
#my_scats = Lb1.curselection()
#start = Lb2.curselection()
#end = Lb3.curselection()
#day = Lb4.curselection()
#direction = Lb5.curselection()
#my_model = Lb6.curselection()
    

    


root.mainloop()