

from math import nan
import os
import time
import datetime as dt
from binance.helpers import date_to_milliseconds
import numpy as np
import json
from numpy import NaN, dtype
from numpy.lib import histograms
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range
import subprocess

def excel_export(df, name='temp_file', size=10000):
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



# init and start the WebSocket
# bsm.start()
# conn = bsm.start_kline_socket(symbol='BTCUSDC', callback=bot.callback, interval = '1m')
# bsm.stop_socket(conn)
pd.read_json

buy_order = client.order_market_buy(symbol='BTCUSDC', quoteOrderQty=round(float(client.get_asset_balance(asset='USDC')['free'])*0.3,0))
# sell_order = client.order_market_sell(symbol='BTCUSDC', quantity = 0.00221)



test_json = dict({'symbol': 'BTCUSDC', 'orderId': 890590059, 'orderListId': -1, 'clientOrderId': '0YICwCqOJQoS6kmOTn88GH', 'transactTime': 1643313808425, 'price': '0.00000000', 'origQty': '0.00066000', 'executedQty': '0.00066000', 'cummulativeQuoteQty': '23.95211940', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'fills': [{'price': '36291.09000000', 'qty': '0.00066000', 'commission': '0.00000066', 'commissionAsset': 'BTC', 'tradeId': 36040801},{'price': '666.09000000', 'qty': '0.00033000', 'commission': '0.00000033', 'commissionAsset': 'BTC', 'tradeId': 36040801}]})

test_json2 = dict({'symbol': 'BTCUSDC', 'orderId': 890590059, 'orderListId': -1, 'clientOrderId': '0YICwCqOJQoS6kmOTn88GH', 'transactTime': 1643313808425, 'price': '0.00000000', 'origQty': '0.00066000', 'executedQty': '0.00066000', 'cummulativeQuoteQty': '23.95211940', 'status': 'FILLED', 'timeInForce': 'GTC', 'type': 'MARKET', 'side': 'BUY', 'fills': [{'price': '36291.09000000', 'qty': '0.00066000', 'commission': '0.00000066', 'commissionAsset': 'BTC', 'tradeId': 36040806}]})

test_json['fills'][1]['price']


dd = pd.json_normalize(test_json['fills'])['price'].apply(pd.to_numeric).mean()





# order = client.get_order(symbol='ETHUSDC',orderId='482480843')
# result = client.cancel_order(symbol='ETHUSDC',orderId='482480843')

# client.get_symbol_info('BTCUSDC')
# client.get_asset_balance('USDC')
# client.get_asset_balance('BTC')


for i in range(1, 4):
    local_time1 = int(time.time() * 1000)
    server_time = client.get_server_time()
    diff1 = server_time['serverTime'] - local_time1
    local_time2 = int(time.time() * 1000)
    diff2 = local_time2 - server_time['serverTime']
    print("local1: %s server:%s local2: %s diff1:%s diff2:%s" % (local_time1, server_time['serverTime'], local_time2, diff1, diff2))
    time.sleep(2)


# bot.socket_error = False
# stop_loop = True
# while True:
# 	# error check to make sure WebSocket is working
#     if bot.socket_error == True:
#         # stop and restart socket
#         bsm.stop_socket(conn)
# 	    #time.sleep(3)
#         conn = bsm.start_kline_socket(symbol='BTCUSDC', callback=bot.callback, interval = '1m')
#         bot.socket_error == False
#     elif stop_loop == True:
#         break 
#     else:
#         None
timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(minutes = 300)).strftime("%d %b %Y %H:%M:%S"))

df = client.get_historical_klines(symbol='BTCUSDT',interval = '1m', start_str = timestamp)


df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['open','high','low','close']].apply(pd.to_numeric)
data = df.copy()

from ta.trend import MACD
macd_obj = MACD(data.close, 15,10,3)
data['MACD'] = macd_obj.macd()
data['MACD_hist'] = macd_obj.macd_diff()
data['MACD_signal'] = macd_obj.macd_signal()

data['position'] = np.where((np.sign(data.MACD)<0) & (data.MACD>data.MACD_signal) & (data.MACD.shift(1)<data.MACD_signal.shift(1)),1,np.nan)
data['position'] = np.where((np.sign(data.MACD)>0) & (data.MACD<data.MACD_signal) & (data.MACD.shift(1)>data.MACD_signal.shift(1)),0,data.position)
data['position']  = data.position.ffill().fillna(0)
data['trades'] = np.where((data.position!=data.position.shift(1)),1,0)
data['trades'] = data.trades.cumsum()-1

excel_export(data)

