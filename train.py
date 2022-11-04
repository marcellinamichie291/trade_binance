import math
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.colab import files


class train():
    def __init__(self, path):
        self.path = path
        
    def train_funct(self):
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = round(len(dataset)*  .85)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:training_data_len , :]
        x_train = []
        y_train = []

        for i in range(60,len(train_data)):
          x_train.append(train_data[i-60:i, 0])
          y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1 )))
        self.model.add(LSTM(50, return_sequences = False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))    

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        callback = tf.keras.callbacks.ModelCheckpoint('model.hdf5', 
                                                  monitor='loss', 
                                                  save_best_only=True, verbose=1)
        self.model.fit(x_train, y_train, batch_size = 32, epochs = 200, callbacks=[callback])
        model = load_model('model.hdf5')
        return model

df = pd.read_csv('btc1h.csv')
x = train(df).train_funct()
files.download('model.hdf5')
