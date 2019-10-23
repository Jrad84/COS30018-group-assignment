"""
Train the NN model.
"""
import sys
import os
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.callbacks import EarlyStopping
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
        
    """
    # Define the Keras TensorBoard callback.
    logdir = os.path.join(
    "logs",
    "fit",
    name,
    'lstm3_FULL',
    datetime.now().strftime("%Y%m%d-%H%M"),
)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.compile(loss="mse", optimizer="adam", metrics=['mape'])
    early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05,
        callbacks=[tensorboard_callback, early])
    
    model.save('model/' + name + '3_layers_FULL'  + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name  +' loss.csv', encoding='utf-8', index=False)

def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(input=p.input,
                                       output=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="adam", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)
    


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()

    lag = 8 # how far to look back
    config = {"batch": 256, "epochs": 20  }
    file1 = './data/train1.csv'
    file2 = './data/test1.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)

    
    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([8, 64, 64, 1]) 
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'simplernn':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_simplernn([8, 64, 64, 1]) 
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([8, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    if args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([8, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv) 
