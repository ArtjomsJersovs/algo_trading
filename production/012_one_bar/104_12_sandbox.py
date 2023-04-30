from math import nan, floor
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
import stored_functions as sf
import telegram_send as ts
ts_conf=r'C:\Users\Administrator\Documents\algo_trading\telegram-send.conf'
# with open("api_keys.json") as file:
#     keys = json.load(file)
# ### API
# binance_api_key = keys['binance_api_key']
# binance_api_secret = keys['binance_api_secret']

client, bsm = sf.setup_api_conn_binance()


start = (pd.to_datetime(dt.datetime.now())-pd.to_timedelta(str('50h'))).strftime('%d %b %Y')
klines = client.get_historical_klines('BTCBUSD', '1h', start)
data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data = data[['open','high','low','close', 'volume']].apply(pd.to_numeric)



x = sf.get_all_binance('BTCBUSD', '1h', save=True, since = pd.to_datetime('2023-04-10').strftime('%d %b %Y'))


x.tail()