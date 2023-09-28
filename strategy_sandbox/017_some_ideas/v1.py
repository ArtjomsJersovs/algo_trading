import os
import time
import datetime as dt
from binance.helpers import date_to_milliseconds, round_step_size
import numpy as np
from numpy import NaN, dtype
from numpy.lib import histograms
import pandas as pd
import unicorn_binance_websocket_api as un
from binance.client import Client
from binance.enums import HistoricalKlinesType
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range, BollingerBands
import ta.momentum as tm
import json
import telegram_send as ts
import matplotlib.pyplot as plt


df = pd.read_csv(r'C:\Users\HP\Documents\GitHub\algo_trading\strategy_sandbox\017_some_ideas\BTCUSDT-trades-2023-09-18.csv')
#df = pd.DataFrame(trades)

#Pervaja ideja - posmotretj standartnoje otklonenije

df['time'] = pd.to_datetime(df['time'], unit='ms')
df.set_index('time', inplace=True)
df['side'] = np.where(df['is_buyer_maker']==False, 1,-1)
df['qty_abs'] = df['qty']
df['qty'] = df['qty']*df['side']
df['counter'] = 1
df = df.resample('5S').agg({'price': 'last', 'qty': 'sum','qty_abs':'sum','counter':'sum'})
indicator_bb = BollingerBands(close=df['price'], window=100, window_dev=2.5)
df['bb_bbh'] = indicator_bb.bollinger_hband()
df['bb_bbl'] = indicator_bb.bollinger_lband()
df['ticks_ma'] = df.counter.rolling(100).mean()
df['sum_qty_ma'] = df.qty_abs.rolling(100).mean()
df['edge_qty_ma'] = df.qty.rolling(100).mean()
df['returns'] = np.log(df.price.astype(float).div(df.price.astype(float).shift(1)))

df['qty_over_ma_qty'] = np.where((df['sum_qty_ma']*40)<df['qty_abs'],50,0)
df.dropna(inplace=True)


ma_period = 100
bb_window = 100
bb_dev = 2.5

def test_strategy_v1(df,ma_period, bb_window, bb_dev):
    indicator_bb = BollingerBands(close=df['price'], window=bb_window, window_dev=bb_dev)
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    df['ticks_ma'] = df.counter.rolling(ma_period).mean()
    df['sum_qty_ma'] = df.qty_abs.rolling(ma_period).mean()
    df['edge_qty_ma'] = df.qty.rolling(ma_period).mean()

    df['over_bbh'] = np.where((df['price'].shift(1)<df['bb_bbh'])&(df['price']>df['bb_bbh']),1,0)
    df['below_bbl'] = np.where((df['price'].shift(1)>df['bb_bbl'])&(df['price']<df['bb_bbl']),1,0)


    #for sell side
    df['position'] = np.where((df['edge_qty_ma']>0)&(df['over_bbh']==1),-1,np.nan)
    df['position_open_price']= abs(df.position*df.price)
    df['position_open_price'] = df.position_open_price.fillna(method='ffill')
    df['position'] = np.where(((df.position_open_price*0.99856)>df.price)&~(np.isnan(df.position_open_price)),0, df.position)
    #df['position'] = np.where(((df['below_bbl']==1)|(df['edge_qty_ma']<0)),0, df.position)
    df['position'] = df['position'].ffill(axis=0).fillna(0)
    #for buy side
    df['position'] = np.where(df['position']!=-1,np.nan, df.position)
    df['position'] = np.where((np.isnan(df['position']))&(df['edge_qty_ma']<0)&(df['below_bbl']==1),1,df.position)
    #df['position'] = np.where((np.isnan(df['position']))&((df['over_bbh']==1)|(df['edge_qty_ma']>0)),0, df.position)
    df['position'] = np.where(((df.position_open_price*1.00144)<df.price)&~(np.isnan(df.position_open_price)),0, df.position)
    df['position'] = df['position'].ffill(axis=0).fillna(0)

    df['strategy'] = df['position'].shift(1) * df['returns']
    df['cumstrategy'] = df['strategy'].cumsum().apply(np.exp)
    df['cumreturns'] = df['returns'].cumsum().apply(np.exp)
    perf = round(df['cumstrategy'].iloc[-1]*100,2)
    benchmark = round(df['cumreturns'].iloc[-1]*100,2)
    df[['cumreturns','cumstrategy']].plot(title = 'strategy X bench', figsize=(12, 8))
    plt.show()
    return perf, benchmark


perf, bench = test_strategy_v1(df,100,100,2)
print(perf)

0.00072*2
1000*1.00144

problema so take profitom - nuzhno tut ponjatj v chem delo i togda nakladyvatj novije filtri, pljus dobavitj kolvo sdeloks