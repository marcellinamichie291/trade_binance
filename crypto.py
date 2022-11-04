#!/usr/bin/env python
# coding: utf-8

import websocket, json, time, datetime, sys, re, os
import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
import math
import tensorflow as tf
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from numpy import loadtxt
from tensorflow import keras
from keras.models import Sequential, load_model
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import pandas as pd
from binance.client import Client
import time
import os
import math
lev = 50


class ML():
    scaler = MinMaxScaler(feature_range=(0,1))
    def __init__(self, stock_data, model_path):
        self.stock_data = stock_data
        self.model = tf.keras.models.load_model(model_path)
        
  
    def pred_values(self):
        a4 = self.stock_data['Close'].values.reshape(-1,1)
        last_60_dayss = a4[-60:]
        last_60_dayss = self.scaler.fit_transform(last_60_dayss)
        next_test = []
        next_test.append(last_60_dayss)
        next_test = np.array(next_test)
        next_test = np.reshape(next_test, (next_test.shape[0], next_test.shape[1], 1))
        price = self.model.predict(next_test)
        pred = self.scaler.inverse_transform(price)
        return float(pred)



api_key = 'your_api_key'
api_secret = 'your_api_secret'
client = Client(api_key, api_secret, testnet=False)




def gethourlydata(symbol, interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback ))
    frame = frame.iloc[:, :6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit = 'ms')
    frame = frame.astype(float)
    return frame




path = '/Users/icarus/Downloads/btc.hdf5' 




def column(df):    
    df['Predictions'] = np.NaN
    for i in range(len(df.values) - 1,len(df.values)):
        df["Predictions"][i] = ML(df.iloc[:i,:], path).pred_values()
    return df

def current_price(pair = 'BTCUSDT'):
    time.sleep(2)
    price = client.get_ticker(symbol = pair)
    ask_price = float(price['askPrice'])
    bid_price = float(price['bidPrice'])
    return ask_price, bid_price
    
pair = 'BTCUSDT'
symbols = client.futures_position_information()
ds = pd.DataFrame(symbols)
symbol_loc = ds.index[ds.symbol == pair]
SYMBOL_POS = (symbol_loc[-1])


class Signals():
    def __init__(self, df):
        self.df = df              
    def decide(self):
        self.df['Buy'] = np.where(self.df['Predictions'] > self.df['Close'] , 1, 0 )
        self.df['Sell'] = np.where(self.df['Predictions'] < self.df['Close'] , -1, 0 )
        
import math
def trun_n_d(n,d):
    dp = repr(n).find('.') #dot position
    if dp == -1:  
        return int(n) 
    return float(repr(n)[:dp+d+1])




def strategy(pair, qty):
    tframe = '5m' #for timeout
    #make the bot sleep if the tp or sl hits 
    time.sleep(600) #for 10 mins
    df = gethourlydata('BTCUSDT', '1h', '3days')
    df = column(df)
    inst = Signals(df)
    inst.decide()
    print(f'current Close is ' + str(df.Close.iloc[-1]))
    print(f'current Predicted is ' + str(df.Predictions.iloc[-1]))
    price = client.get_ticker(symbol = pair)
    ask_price = float(price['askPrice'])
    bid_price = float(price['bidPrice'])
    target = float(df.Predictions.iloc[-1])
    per_buy =  (target-ask_price)/ask_price
    per_sell = (target-bid_price)/bid_price 
    print('buy_change is '+ str(per_buy))
    print('sell_change is ' + str(per_sell))

    if tframe[-1] == 'm':
        tf1 = int(re.findall('\d+', tframe)[0])
        tme_frame = 1 * tf1
    if tframe[-1] == 'h':
        tf1 = int(re.findall('\d+', tframe)[0])
        tme_frame = 60 * tf1
    
    if per_buy >= 0.01 and df.Buy.iloc[-1]:
        #check position if already exist
        SYMBOL_POS = 'BTCUSDT'
        check_if_in_position = client.futures_position_information()
        position = pd.DataFrame(check_if_in_position)
        pos = (position.loc[position['symbol'] == 'BTCUSDT'])
        position_amount = float(pos['positionAmt'])

        #if not in position will proceed to buy
        if position_amount == 0:
            print("cancelling all previous sl or tp orders before opening new position")
            cancel_orders = client.futures_cancel_all_open_orders(symbol = pair)
            print('position already does not exit, so executing order')
            entry_price = float(ask_price)
            print("Entry Price at: {}".format(entry_price))
            now_balance = client.futures_account_balance()
            dt = pd.DataFrame(now_balance)
            a = float(dt.loc[dt['asset']=='USDT']['balance'])
            pos_size = ask_price * qty 
            account_bal = a
            exposure = pos_size / account_bal 
            percentt = 1 - (0.03/exposure)
            stop_losss = percentt * entry_price
            stop_loss = trun_n_d(stop_losss,2)
            print("Calculated stop loss at " + str(stop_loss))
            take_profitt = float(df.Predictions.iloc[-1])
            take_profit = trun_n_d(take_profitt,2)
            print("Calculated take profit at " + str(take_profit))

            try:
                buy_limit_order = client.futures_create_order(symbol=pair, side='BUY', type='MARKET', quantity=qty)
                order_id = buy_limit_order['orderId']
                order_status = buy_limit_order['status']

                timeout = time.time() + (50 * tme_frame)
                while order_status != 'FILLED':
                    time.sleep(10)
                    order_status = client.futures_get_order(symbol=pair, orderId=order_id)['status']
                    print(order_status)

                    if order_status == 'FILLED':
                        time.sleep(1)
                        set_stop_loss = client.futures_create_order(symbol=pair, side='SELL', type='STOP_MARKET', quantity=qty, stopPrice=stop_loss)
                        time.sleep(1)
                        set_take_profit = client.futures_create_order(symbol=pair, side='SELL', type='TAKE_PROFIT_MARKET', quantity=qty, stopPrice=take_profit)
                        break

                    if time.time() > timeout:
                        order_status = client.futures_get_order(symbol=pair, orderId=order_id)['status']
                        
                        if order_status == 'PARTIALLY_FILLED':
                            cancel_order = client.futures_cancel_order(symbol=pair, orderId=order_id)
                            time.sleep(1)
                            
                            pos_size = client.futures_position_information()
                            d_size = pd.DataFrame(pos_size)
                            pos = (d_size.loc[d_size['symbol'] == 'BTCUSDT'])
                            pos_amount = pos['positionAmt']
                            print('The current position amount is ' + str(pos_amount))

                            time.sleep(1)
                            set_stop_loss = client.futures_create_order(symbol=pair, side='SELL', type='STOP_MARKET', quantity=pos_amount, stopPrice=stop_loss)
                            time.sleep(1)
                            set_take_profit = client.futures_create_order(symbol=pair, side='SELL', type='TAKE_PROFIT_MARKET', quantity=pos_amount, stopPrice=take_profit)
                            break

                        else:
                            cancel_order = client.futures_cancel_order(symbol=pair, orderId=order_id)
                            break
            except BinanceAPIException as e:
                # error handling goes here
                print(e)
            except BinanceOrderException as e:
                # error handling goes here
                print(e)
                    

                    
    if per_sell <= -0.01 and df.Sell.iloc[-1]:
            check_if_in_position = client.futures_position_information()
            d_pos = pd.DataFrame(check_if_in_position)
            pos = (d_pos.loc[d_pos['symbol'] == 'BTCUSDT'])
            position_amount = float(pos['positionAmt'])
                
            if float(position_amount) == 0:
                print("cancelling all previous sl or tp orders before opening new position")
                cancel_orders = client.futures_cancel_all_open_orders(symbol = pair)
                print('SELL SIGNAL IS ON! Executing order')
                entry_price = float(bid_price)
                print("Entry Price at: {}".format(entry_price))
                now_balance = client.futures_account_balance()
                dt = pd.DataFrame(now_balance)
                a = float(dt.loc[dt['asset']=='USDT']['balance'])
                account_bal = a
                pos_size = bid_price * qty 
                exposure = pos_size / account_bal 
                percentt = (0.03/exposure) + 1 
                stop_losss = percentt * entry_price
                stop_loss = trun_n_d(stop_losss,2)

                print("Calculated stop loss at: {}".format(stop_loss))

                take_profit = float(df.Predictions.iloc[-1])
                take_profit = trun_n_d(take_profit,2)
                print("Calculated take profit at: {}".format(take_profit))


                try:
                    sell_limit_order = client.futures_create_order(symbol=pair, side='SELL', type='MARKET', quantity=qty)
                    order_id = sell_limit_order['orderId']
                    order_status = sell_limit_order['status']

                    timeout = time.time() + (50 * tme_frame)
                    while order_status != 'FILLED':
                        time.sleep(10) #check every 10sec if limit order has been filled
                        order_status = client.futures_get_order(symbol=pair, orderId=order_id)['status']
                        print(order_status)

                        if order_status == 'FILLED':
                            time.sleep(1)
                            set_stop_loss = client.futures_create_order(symbol=pair, side='BUY', type='STOP_MARKET', quantity=qty, stopPrice=stop_loss)
                            time.sleep(1)
                            set_take_profit = client.futures_create_order(symbol=pair, side='BUY', type='TAKE_PROFIT_MARKET', quantity=qty, stopPrice=take_profit)
                            break

                        if time.time() > timeout:
                            order_status = client.futures_get_order(symbol=pair, orderId=order_id)['status']
                            
                            if order_status == 'PARTIALLY_FILLED':
                                cancel_order = client.futures_cancel_order(symbol=pair, orderId=order_id)
                                time.sleep(1)
                                
                                pos_size = client.futures_position_information()
                                d_sizee = pd.DataFrame(pos_size)
                                pos = (d_sizee.loc[d_sizee['symbol'] == 'BTCUSDT'])
                                print('Your partial position size filled is ' + str(pos))

                                time.sleep(1)
                                set_stop_loss = client.futures_create_order(symbol=pair, side='BUY', type='STOP_MARKET', quantity=qty, stopPrice=stop_loss)
                                time.sleep(1)
                                set_take_profit = client.futures_create_order(symbol=pair, side='BUY', type='TAKE_PROFIT_MARKET', quantity=qty, stopPrice=take_profit)
                                break
                            
                            else:
                                cancel_order = client.futures_cancel_order(symbol=pair, orderId=order_id)
                                break

                except BinanceAPIException as e:
                    # error handling goes here
                    print(e)
                except BinanceOrderException as e:
                    # error handling goes here
                    print(e)




while True:
        strategy('BTCUSDT', 0.001)
