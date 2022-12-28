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
from binance.enums import HistoricalKlinesType
from pandas.core.tools.datetimes import to_datetime
from ta.trend import MACD
import json
import telegram_send as ts
import stored_functions as sf

### API
from binance import ThreadedWebsocketManager
from binance.client import Client
with open("api_keys.json") as file:
    credentials = json.load(file)

binance_api_key = credentials['binance_api_key'] 
binance_api_secret = credentials['binance_api_secret']

ts_conf='telegram-send.conf'
### CONSTANTS

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
bsm = ThreadedWebsocketManager()
info = client.futures_account_balance()[-1]['balance']
print('Connection to Binance established! Account data: {}'.format(info))

timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(minutes = 60)).strftime("%d %b %Y %H:%M:%S"))

price = client.futures_symbol_ticker(symbol='BTCBUSD')

df = pd.DataFrame(client.futures_historical_klines(
    symbol='BTCBUSD',
    interval='5m',
    start_str=timestamp
))
df.head()


leverage = 2
client.futures_change_leverage(symbol='BTCBUSD', leverage=leverage)

test_order_buy = client.futures_create_order(
    symbol='BTCBUSD',
    type='MARKET',
    side='BUY',
    newOrderRespType = 'RESULT',
    quantity=round_step_size((floor(float(info)/10)*10)*leverage/float(price['price']), 0.001)
)

client.futures_get_open_orders(symbol='BTCBUSD')

float(client.futures_get_all_orders(symbol='BTCBUSD')[-1]['origQty']) #take last order

test_order_sell = client.futures_create_order(
    symbol='BTCBUSD',
    #price = 19300,
    #timeInForce = 'GTC',
    type='MARKET',
    side='SELL',
    newOrderRespType = 'RESULT',
    quantity=float(client.futures_get_all_orders(symbol='BTCBUSD')[-1]['origQty'])
)

client.futures_account_trades(symbol='BTCBUSD')[-3]
test = client.futures_get_order(symbol='BTCBUSD',orderId=16708907326)
#client.futures_cancel_all_open_orders()

#check out the slippage and sum all orders - simulate such situation 
#pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum()
pd.json_normalize(test['cumQuote'])


df = self.data.apply(pd.to_numeric).copy()
#df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))

#******************** define your strategy here ************************
def get_historical_data():
    timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(minutes = 200)).strftime("%d %b %Y %H:%M:%S"))
    df = client.get_historical_klines(symbol='BTCUSDT',interval = '5m', start_str = timestamp, klines_type=HistoricalKlinesType.FUTURES)
    df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open','high','low','close', 'volume']].apply(pd.to_numeric)
    df['skip_data'] = 1
    return df

df = get_historical_data()
from ta.volatility import average_true_range
df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.close, window = 18, fillna=True)
df['RSI'] = tm.rsi(df.close, self.lookback_bars, fillna=True)
df['volume_ma'] = self.data['volume'].rolling(self.ma_interval).mean()
df[f'max_price_{self.lookback_bars}'] = self.data['high'].rolling(self.lookback_bars).max()
df[f'min_price_{self.lookback_bars}'] = self.data['low'].rolling(self.lookback_bars).min()

df['breaks_local_high'] = (df[f'max_price_{self.lookback_bars}'].shift(1)<=df.close) & (df[f'max_price_{self.lookback_bars}']>df.close.shift(1))
df['breaks_local_low'] = (df[f'min_price_{self.lookback_bars}'].shift(1)>=df.close) & (df[f'min_price_{self.lookback_bars}']<df.close.shift(1))
df['RSI_neutral'] = (df.RSI >= 40) & (df.RSI <=60)
    
35*5



test = '{"id":2,"result":null}'
test['id']