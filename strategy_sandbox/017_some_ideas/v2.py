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
import itertools
pd.set_option('display.max_rows', 100)

df = pd.read_csv(r'C:\Users\HP\Documents\GitHub\algo_trading\strategy_sandbox\017_some_ideas\ETHUSDT-trades-2023-08.csv')
df['time'] = pd.to_datetime(df['time'], unit='ms')
df.set_index('time', inplace=True)
df['side'] = np.where(df['is_buyer_maker']==False, 1,-1)
df['qty_abs'] = df['qty']
df['qty'] = df['qty']*df['side']
df['counter'] = 1
df = df.resample('5S').agg({'price': 'last', 'qty': 'sum','qty_abs':'sum','counter':'sum'})
df['returns'] = np.log(df.price.astype(float).div(df.price.astype(float).shift(1)))

#df_test = df.loc['2023-09-18 16:30:00.00':'2023-09-18 18:30:00.00']
#df_test = df
# df['edge_qty_ma'] = df.qty.rolling(100).sum()
# df['edge_qty_short_ma'] = df.qty.rolling(10).sum()
# df['edge_ratio'] = df['edge_qty_short_ma']/df['edge_qty_ma']

long_ma=100
short_ma=10
cutoff=1000
commission=0.00036
size=1000

df_clean = df.copy()
df = df_clean.copy()

def test_strategy_v1(df,long_ma=100, short_ma=10, cutoff=1000, commission=0.00036, size=1000):
    #for buy side
    df['position'] = np.where((df.edge_ratio>cutoff),1,np.nan)
    df['trade'] = np.where(df.position==1,1,0)
    df['position_open_price']= abs(df.position*df.price)
    df['position_open_price'] = df.position_open_price.fillna(method='ffill')
    df['position'] = np.where((((df.position_open_price*1.00144)<df.price)|((df.position_open_price*0.99856)>df.price))&(df.position.shift(1)!=0),0, df.position)
    df['position'] = df.position.ffill(axis=0).fillna(0)
    #for sell side
    df['position'] = np.where(df.position==0,np.nan,df.position)
    df['position'] = np.where((df.edge_ratio<(cutoff*(-1)))&(df.position!=1),-1,df.position)
    df['trade'] = np.where(df.position==-1,1,df.trade)
    df['position_open_price'] = np.where(df.position!=1,np.nan, df.position_open_price)
    df['position_open_price'] = abs(df.position*df.price)
    df['position_open_price'] = df.position_open_price.fillna(method='ffill')
    df['position'] = np.where((((df.position_open_price*1.00144)<df.price)|((df.position_open_price*0.99856)>df.price))&(df.position.shift(1)!=0),0, df.position)
    df['position'] = df.position.ffill(axis=0).fillna(0)

    df['strategy'] = df['position'].shift(1) * df['returns']
    df['cumstrategy'] = df['strategy'].cumsum().apply(np.exp)
    df['cumreturns'] = df['returns'].cumsum().apply(np.exp)
    df['last_record_position'] = np.where((df.position.shift(1)!=0)&(df.position==0),df.position.shift(1),0)
    df['pos_result'] = np.where((df.position_open_price<=df.price)&(df.last_record_position==1),1,
                                np.where((df.position_open_price>df.price)&(df.last_record_position==1),0,
                                np.where((df.position_open_price>=df.price)&(df.last_record_position==-1),1,
                                np.where((df.position_open_price<df.price)&(df.last_record_position==-1),0,np.nan))))
    perf = round(df['cumstrategy'].iloc[-1]*100,2)
    benchmark = round(df['cumreturns'].iloc[-1]*100,2)
    trades = df['trade'].sum()
    comm = commission * size * trades * 2
    pnl = size * (perf/100) - comm 
    acc = df['pos_result'].sum()/trades
    #df[['cumreturns','cumstrategy']].plot(title = 'strategy X bench', figsize=(12, 8))
    #plt.show()
    results = {'perf':perf,'benchmark':benchmark,'trades':trades, 'comm':comm, 'PnL':pnl, 'acc':acc}
    return results



df_res, res = test_strategy_v1(df,long_ma=100, short_ma=10, cutoff=1000, commission=0.00036, size=1000)

df.head(100000).to_csv(R'C:\Users\HP\Documents\GitHub\algo_trading\strategy_sandbox\017_some_ideas\test.csv')

df['pos_buy'] = np.where(df['position']==1,1,0)
df['pos_sell'] = np.where(df['position']==-1,1,0)

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(df.index, df['price'], label='Price', color='blue')
ax1.plot(df.index, df['price'].rolling(1000).mean(), label='Price', color='orange')
ax2 = ax1.twinx()
ax2.plot(df.index, df['pos_buy'], label='ticks', color='green', alpha=0.7)
ax2.plot(df.index, df['pos_sell'], label='ticks', color='red', alpha=0.7)
#ax2.plot(df_test.index, df_test['btc_edge_sign'], label='BTC Price', color='red', alpha=0.7)
plt.show()


df = df_clean.copy()
df['edge_qty_ma'] = df.qty.rolling(long_ma).sum()
df['edge_qty_short_ma'] = df.qty.rolling(short_ma).sum()
df['edge_ratio'] = df['edge_qty_short_ma']/df['edge_qty_ma']
df.dropna(inplace=True)

#Iterator of best params tf and pair
long_ma_int = list(range(50,550,50))
short_ma_int = list(range(5,55,5))
cutoff_int = list(range(600,2200,200))

all_combinations = list(itertools.product(long_ma_int, short_ma_int, cutoff_int))
data_to_append = []

#df = pd.DataFrame(columns =['combination' ,'perf','benchmark','trades', 'comm','PnL', 'acc'])
start_time = time.time()
counter = 1
for i in all_combinations:
    res = test_strategy_v1(df,long_ma=i[0], short_ma=i[1], cutoff=i[2], commission=0.00036, size=1000)
# Create a dictionary for the row you want to append
    # Append the row data dictionary to the list
    data_to_append.append(res)

    # Convert the list of dictionaries into a DataFrame
    #df = pd.concat([df, pd.DataFrame(data_to_append)], ignore_index=True)

    print(f'{counter} / {len(all_combinations)} done...')
    counter += 1

results = pd.DataFrame(data_to_append)
results = pd.concat([results, pd.DataFrame(all_combinations, columns=['long_ma','short_ma','cutoff'])], axis=1)
results.to_csv('C:/Users/HP/Documents/GitHub/algo_trading/strategy_sandbox/017_some_ideas/test_results.csv')
opt_acc = results.iloc[np.argmax(results.acc)]
opt_perf = results.iloc[np.argmax(results.perf)]
print(75*"-")
print('The Best performance is: \n{}'.format(opt_perf))
print('The Best accuracy is: \n{}'.format(opt_acc))
print(f'time spent on gridsearch in minutes: {round((time.time() - start_time)/60,2)}')




