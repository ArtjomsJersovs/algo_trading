

from audioop import avg
from math import nan
import os
import time
import json
import datetime as dt
from binance.helpers import date_to_milliseconds
import numpy as np
from numpy import NaN, dtype, std
from numpy.lib import histograms
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range
from ta import add_all_ta_features
from stored_functions import excel_export
import subprocess
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def excel_export(df, name='temp_file', size=1000000):
    df.head(int(size)).to_excel(str(name) +".xlsx") 
    subprocess.run(["C:/Program Files/Microsoft Office/root/Office16/EXCEL.exe", str(name) +".xlsx"])


pd.set_option('display.max_rows', 100)
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

#GET DATA

tf = pd.to_timedelta('5m'.replace('m','min'))
filename = str(os.getcwd())+'\\datasets\\%s-1m-data.csv' % ('BTCUSDT')
raw = pd.read_csv(filename)
raw['timestamp'] = pd.to_datetime(raw['timestamp'])
raw.set_index('timestamp', inplace=True)
raw = raw.resample(tf, label = 'right').last().ffill().apply(pd.to_numeric)

#GET ALL TECHNICAL INDICATOR DATA
raw_ta = add_all_ta_features(raw,open='open',high='high',low='low',close='close', volume='volume')
raw_ta_cut = raw_ta.iloc[45:,]

#FILTER OUT ONLY RELATIVE ONES
cols = ["open","high","low","close","volume","volatility_bbp","volatility_bbhi","volatility_bbli","volatility_kchi","volatility_kcli","volatility_dcl","trend_macd","trend_macd_signal","trend_macd_diff","trend_adx_pos","trend_adx_neg","trend_trix","trend_mass_index","trend_cci","trend_dpo","trend_kst","trend_kst_sig","trend_aroon_up","trend_aroon_down","trend_aroon_ind","trend_psar_up_indicator","trend_psar_down_indicator","momentum_rsi","momentum_stoch_rsi","momentum_stoch_rsi_k","momentum_stoch_rsi_d","momentum_tsi","momentum_uo","momentum_stoch","momentum_stoch_signal","momentum_wr","momentum_ao","momentum_ppo","momentum_ppo_signal","momentum_ppo_hist","others_dr","others_dlr","others_cr"]


#CUT AND PREPROCESS DATASET
raw_ta_cut_t = raw_ta_cut.loc[:,cols]
df = raw_ta_cut_t.apply(pd.to_numeric).copy()
df['returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
#df.returns.describe()
#PLOT MEAN RETURNS
 #df.returns.loc[df['returns']>-0.1,].hist(bins=200)
 #plt.show()
 #df.returns.mean()*1440

df = df['2022']
df = df.iloc[1:,]
df['time'] = df.index
df['time'] = pd.to_datetime(df['time'])
df['dayofweek'] = df['time'].dt.day_of_week
df['hour'] = df['time'].dt.hour
df = df.drop(['time', ''], axis=1)
df.to_csv('004_xgboost_model/BTCUSDT_5m_2022_proc.csv')


