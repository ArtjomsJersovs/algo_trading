import os
import numpy as np
import pandas as pd
import binance as bin
import requests 
import json 
import datetime as dt
import matplotlib.pyplot as plt
import stored_functions as sf

plt.style.use("seaborn")


df = pd.read_csv('C:/Python_Scripts/datasets/BTCUSDT_15m_full.csv')
df.timestamp = pd.to_datetime(df.timestamp)
df.close_time = df.timestamp + dt.timedelta(minutes=15)
df = df.set_index(df.close_time).drop("close_time", axis=1)
main_df = df.copy().dropna().apply(pd.to_numeric)

### some preprocessing features
#df.resample("MS", loffset="14D").mean()
float(0.01) / df['close'].mean()
close = df.resample("D").last().loc[:,"close"].copy()
close.plot(figsize = (15,8),fontsize = 13)
plt.legend(fontsize = 15)
# plt.show()

# close.div(close.iloc[0]).mul(100)
# close.shift(periods =-1)
df['Lag1'] = df.close.shift(periods=1)
df['pct_change'] = df.close.div(df.Lag1).sub(1).mul(100)
#sf.excel_export(df,'testings')


from ta import add_all_ta_features
from ta.volatility import average_true_range

atr_period = 30
multiplicator = 2.1
test_df = add_all_ta_features(main_df, open="open", high="high", low="low", close="close", volume="volume")

main_df['ATR'] = average_true_range(high = main_df.high, low = main_df.low, close = main_df.high, window = atr_period)
main_df = main_df[main_df.ATR!= 0]


main_df['position'] = np.where((main_df.close-main_df.open)> main_df['ATR']*multiplicator, 1, np.nan)
main_df['position_open_price'] = np.where((main_df.close-main_df.open)> main_df['ATR']*multiplicator, main_df.close, np.nan)
main_df['position_open_price'] = main_df['position_open_price'].ffill(axis=0)
main_df['position'] = np.where((main_df['close'] - main_df['position_open_price']>main_df['ATR']*5), 0, main_df['position'])
main_df['position'] = np.where(main_df['close'] < main_df['position_open_price'] - main_df['ATR']*(3), 0, main_df['position'])
main_df['position'] = main_df['position'].ffill(axis=0).fillna(0)

sf.excel_export(main_df.loc['2021-05-01':],'test', size = 10000)




sma_s = 50
sma_l = 200
position = 0
main_df["SMA_S"] = main_df.close.rolling(sma_s).mean()
main_df["SMA_L"] = main_df.close.rolling(sma_l).mean()

for bar in range(len(main_df)):
    if main_df['SMA_S'].iloc[bar] > main_df['SMA_L'].iloc[bar]:
        if position in [0,-1]:
            print('{} Go long | Price {} '.format(main_df.index[bar].date(),main_df.close.iloc[bar]))
            position = 1 
    elif main_df['SMA_S'].iloc[bar] < main_df['SMA_L'].iloc[bar]:
        if position in [0,1]:
            print('{} Go short | Price {} '.format(main_df.index[bar].date(),main_df.close.iloc[bar]))
            position = -1 
