from math import nan, floor
import os
import time
import datetime as dt
from binance.helpers import date_to_milliseconds, round_step_size
import numpy as np
from numpy import NaN, dtype
from numpy.lib import histograms
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range
from ta.momentum import RSIIndicator
from ta.trend import MACD
import json
import subprocess
import matplotlib.pyplot as plt
import os.path
import time
from datetime import timedelta, datetime
from dateutil import parser


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)

# init
### API
with open("api_keys.json") as file:
    credentials = json.load(file)

binance_api_key = credentials['binance_api_key'] 
binance_api_secret = credentials['binance_api_secret']

ts_conf='telegram-send.conf'
### CONSTANTS

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

bsm = ThreadedWebsocketManager()
timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(minutes = 1000)).strftime("%d %b %Y %H:%M:%S"))
df = client.get_historical_klines(symbol='BTCUSDT',interval = '1m', start_str = timestamp)
df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['open','high','low','close']].apply(pd.to_numeric)



def excel_export(df, name='temp_file', size=10000):
    df.head(int(size)).to_excel(str(name) +".xlsx") 
    subprocess.run(["C:/Program Files/Microsoft Office/root/Office16/EXCEL.exe", str(name) +".xlsx"])



#******ADD NECESSARY INDICATORS*************
#******************** define your strategy here ************************
df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))

#******************** define your strategy here ************************
macd_obj = MACD(df.close, 15,6,3)
rsi_obj = RSIIndicator(df.close, 28)
df['MACD'] = macd_obj.macd()
df['MACD_hist'] = macd_obj.macd_diff()
df['MACD_signal'] = macd_obj.macd_signal()
df['RSI'] = rsi_obj.rsi()
df['position'] = np.where((df.MACD>0) & (np.sign(df.MACD_hist)>np.sign(df.MACD_hist.shift(1))), 1, np.nan)
df['position'] = np.where((np.sign(df.MACD_hist)<np.sign(df.MACD_hist.shift(1))), 0, df.position)
df['position']  = df.position.ffill().fillna(0)
df['strategy_returns'] = df['position'].shift(1) * df['bench_returns']
df['cum_strategy_returns'] = df['strategy_returns'].cumsum().apply(np.exp)


df.close.shift(1)[1]

