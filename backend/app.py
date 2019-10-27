# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:16:12 2019

@author: User
"""

from flask import Flask
app = Flask(__name__)
@app.route("/")
def index():
    return "Hello World!"
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8000)