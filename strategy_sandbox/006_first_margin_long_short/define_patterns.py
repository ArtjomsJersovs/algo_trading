import os
from pydoc import describe
import time
import datetime as dt
from binance.helpers import date_to_milliseconds
import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range
import stored_functions as sf
import subprocess
import matplotlib.pyplot as plt
from matplotlib import pyplot
from tqdm import tqdm
plt.style.use("ggplot")


pd.set_option('display.max_rows', 100)
#INIT API
client, bsm = sf.setup_api_conn_binance()


symbol = 'BTCUSDT'
tf = '1m'

filename = str(os.getcwd())+'\\strategy_sandbox\\datasets\\%s-%s-data.csv' % (symbol, tf)
if os.path.isfile(filename):
    raw = pd.read_csv(filename)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    raw.set_index('timestamp', inplace=True)
    #raw = raw.resample(tf, label = 'right').last().ffill().apply(pd.to_numeric)
    print('All caught up! Existing dataset is used.')
else:
 raw = sf.get_all_binance(symbol, tf, save=True)

#iterate the strategy
lookback_bars = 100
ma_interval = 20
body_size = 0.30
extremum_ratio_size = 0.40

raw['ATR'] = average_true_range(high = raw.high, low = raw.low, close = raw.close, window = ma_interval)
raw['volume_ma'] = raw['volume'].rolling(ma_interval).mean()
# Add criteria for high /low of previous data
raw[f'max_price_{lookback_bars}'] = raw['high'].rolling(lookback_bars).max()
raw[f'min_price_{lookback_bars}'] = raw['low'].rolling(lookback_bars).min()

raw['position'] = np.nan
raw['rng'] = np.nan
raw['ratio_body'] = np.nan
raw['ratio_low'] = np.nan
raw['ratio_high'] = np.nan
raw['rng_to_atr'] = np.nan


raw = raw['2022-06-01':]

for i in tqdm(range(len(raw)-1)):
    #formulate pinbar appearance
    rng = abs(raw.high.iloc[i]-raw.low.iloc[i])
    open_to_close = abs(raw.close.iloc[i]-raw.open.iloc[i])
    ratio_body = open_to_close/rng
    
    close_to_low = abs(raw.close.iloc[i]-raw.low.iloc[i])
    close_to_high = abs(raw.close.iloc[i]-raw.high.iloc[i])
    ratio_low = close_to_low/rng
    ratio_high = close_to_high/rng
    
    #flags
    ratio_body_flag = ratio_body < body_size
    ratio_low_flag = ratio_low <=0.40
    ratio_high_flag = ratio_high <=0.40
    over_atr_flag = rng > (raw.ATR.iloc[i]*2)
    over_vol_flag = raw.volume.iloc[i] > raw.volume_ma.iloc[i]
    around_prev_max_flag = (raw[f'max_price_{lookback_bars}'].iloc[i]>raw.close.iloc[i]) and (raw[f'max_price_{lookback_bars}'].iloc[i]<raw.high.iloc[i])
    around_prev_min_flag = (raw[f'min_price_{lookback_bars}'].iloc[i]<raw.close.iloc[i]) and (raw[f'min_price_{lookback_bars}'].iloc[i]>raw.low.iloc[i])

    
    if ratio_body_flag and ratio_low_flag and over_atr_flag and over_vol_flag:# and around_prev_max_flag:
        raw['position'].iloc[i] = -1
    elif ratio_body_flag and ratio_high_flag and over_atr_flag and over_vol_flag:# and around_prev_min_flag:
        raw['position'].iloc[i] = 1
    else:
        raw['position'].iloc[i] = 0
        
print('done')

print(raw.loc[raw['position']!=0])
        
sf.excel_export(raw['2022-09-10':]) 

