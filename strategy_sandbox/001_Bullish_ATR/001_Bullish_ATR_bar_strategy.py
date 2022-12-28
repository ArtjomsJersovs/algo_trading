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
import json

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
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

class Trader():

    def __init__(self, ticker, interval, hist_period_minutes, atr_period, multiplicator, TP_atr=5, SL_atr=3, units=0):
        self.ticker = ticker
        self.hist_period_minutes = hist_period_minutes
        self.tick_data = pd.DataFrame()
        self.interval = pd.to_timedelta(str(interval).replace('m','min'))
        self.interval_str = interval
        self.buy_order = dict()
        self.sell_order = dict()
        self.units = units
        self.executed_qty = 0
        self.position = 0
        self.bnb_price = 0
        self.last_bar = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
        self.data = self.get_historical_data().assign(order_price=np.nan, order_qty=np.nan,commission=np.nan)
        self.socket_error = False

        #*****************add strategy-specific attributes here******************
        self.atr_period = atr_period
        self.multiplicator = multiplicator
        self.TP_atr = TP_atr
        self.SL_atr = SL_atr
        #************************************************************************
        self.define_strategy()
        self.trades_data = pd.DataFrame(columns = ['raw_order_response', 'bnb_topup'])

    def callback(self, df):
        global conn 
        recent_tick = pd.Timestamp.utcnow().tz_localize(None)

        # IF receives API error then it will restart the connection
        if df['e'] == 'error':
          bsm.stop_socket(conn)
          print('Socket stopped')
          time.sleep(3)
          conn = bsm.start_kline_socket(symbol=self.ticker, callback=self.callback, interval = self.interval_str)


        if (recent_tick - self.last_bar) > self.interval:
            print("{} at {}".format(round(float(df['k']['c']),0),recent_tick))
            
        df = pd.DataFrame({"open":df['k']['o'],
                        "high":df['k']['h'],
                        "low":df['k']['l'],
                        "close":df['k']['c']},
                        index = [pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))])
        self.tick_data = self.tick_data.append(df)

        if recent_tick - self.last_bar > self.interval:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()


    def get_historical_data(self):
        timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(minutes = 60)).strftime("%d %b %Y %H:%M:%S"))
        df = client.get_historical_klines(symbol='BTCUSDT',interval = self.interval_str, start_str = timestamp)
        df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open','high','low','close']].apply(pd.to_numeric)
        df['skip_data'] = 1
        return df
  
    def resample_and_join(self):
        self.data = self.data.append(self.tick_data.resample(self.interval, label = 'right').last().ffill().apply(pd.to_numeric).iloc[:-1])  #ne bratj poslednij, chtoby v data byli toljko polnije bary
        self.data = self.data.iloc[-self.hist_period_minutes:] #bratj toljko poslednije 300 barov
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.data.index[-1]

    def define_strategy(self): # "strategy-specific"
        df = self.data.apply(pd.to_numeric).copy()
        df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))

        #******************** define your strategy here ************************
        df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.close, window = self.atr_period)
        df['position'] = np.where((df.close-df.open> df.ATR*self.multiplicator) & (df.skip_data != 1) , 1, np.nan)
        df['position_open_price']= np.where(len(df) == 0, 0, df['close'].iloc[-1])
        df['position'] = np.where((df.close - df.position_open_price>df.ATR*self.TP_atr) & (df.position==1), 0, df.position)
        df['position'] = np.where((df.close < df.position_open_price - df.ATR*self.SL_atr) & (df.position==1), 0, df.position)
        df['position'] = df.position.ffill(axis=0).fillna(0)

        df['strategy_returns'] = df['position'].shift(1) * df['bench_returns']
        df['cum_bench_returns'] = df['bench_returns'].cumsum().apply(np.exp)
        df['cum_strategy_returns'] = df['strategy_returns'].cumsum().apply(np.exp)
        
        #***********************************************************************
        self.data = df.copy()

    def execute_trades(self):

        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:

            #Main order
                self.buy_order = client.order_market_buy(symbol=self.ticker, quantity=round_step_size(floor(float(client.get_asset_balance(asset='USDC')['free'])*0.5)/float(self.tick_data['close'].iloc[-1]), 0.00001))

            #Technical fields
                self.bnb_price = float(client.get_symbol_ticker(symbol="BNBUSDC")['price'])
                self.data['order_price'].iloc[-1]= pd.json_normalize(self.buy_order['fills'])['price'].apply(pd.to_numeric).mean()
                self.data['order_qty'].iloc[-1] = pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum()
                self.data['commission'].iloc[-1] = pd.json_normalize(self.buy_order['fills'])['commission'].apply(pd.to_numeric).sum()*self.bnb_price
                self.data['cum_commission_usd'] = self.data['commission'].cumsum()
 
                self.topup_bnb()
                self.trades_data =self.trades_data.append(self.data.iloc[-1])
                self.trades_data['raw_order_response'].iloc[-1] = str(self.buy_order) 
                self.report_trade(self.buy_order, "GOING LONG")
                file_name = 'C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/001_Bullish_ATR_bar_tradelog_{}.csv'.format(pd.Timestamp.utcnow().tz_localize(None).strftime("%Y%m%d"))
                self.trades_data.to_csv(file_name)
            self.position = 1
            
        elif self.data["position"].iloc[-1] == 0:
            if self.position == 1: 

            #Main order  
                self.sell_order = client.order_market_sell(symbol=self.ticker, quantity = pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum())

            #Technical fields
                self.bnb_price = float(client.get_symbol_ticker(symbol="BNBUSDC")['price'])
                self.data['order_price'].iloc[-1]= pd.json_normalize(self.sell_order['fills'])['price'].apply(pd.to_numeric).mean()
                self.data['order_qty'].iloc[-1] = pd.json_normalize(self.sell_order['fills'])['qty'].apply(pd.to_numeric).sum()
                self.data['commission'].iloc[-1] = pd.json_normalize(self.sell_order['fills'])['commission'].apply(pd.to_numeric).sum()*self.bnb_price
                self.data['cum_commission_usd'] = self.data['commission'].cumsum()
                
                self.topup_bnb()
                self.trades_data =self.trades_data.append(self.data.iloc[-1])
                self.trades_data['raw_order_response'].iloc[-1] = str(self.sell_order) 
                self.report_trade(self.sell_order, "GOING NEUTRAL")
                file_name = 'C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/001_Bullish_ATR_bar_tradelog_{}.csv'.format(pd.Timestamp.utcnow().tz_localize(None).strftime("%Y%m%d"))
                self.trades_data.to_csv(file_name)
            self.position = 0

        
    def report_trade(self, order, text):
        print("\n" + 100* "-")
        print(text)
        print("Filled quantity: {} - at price: {}".format(order['fills'][0]['qty'],order['fills'][0]['price']))
        print('Cumulative strategy returns = {}'.format(self.data['cum_strategy_returns'].iloc[-1]))
        print(100 * "-" + "\n")  


    def topup_bnb(self):
        ''' Top up BNB balance if it drops below minimum specified balance '''
        bnb_balance = float(client.get_asset_balance(asset='BNB')['free'])
        usd_balance = bnb_balance * self.bnb_price
        if usd_balance < 10:
            order = client.order_market_buy(symbol='BNBUSDC', quoteOrderQty=10)
            self.data['bnb_topup'].iloc[-1] = str(order)
        else: pass





#if __name__ == "__main__":
    
#bot = Trader(ticker = 'BTCUSDC',interval='15m', hist_period_minutes=100,  atr_period = 19, multiplicator=2.3, TP_atr=2.8, SL_atr=2.8) real one

bot = Trader(ticker = 'BTCUSDC',interval='1m', hist_period_minutes=1600,  atr_period = 10, multiplicator=1.1, TP_atr=3, SL_atr=1.5)

# init and start the WebSocket
bsm.start()
conn = bsm.start_kline_socket(symbol=bot.ticker, callback=bot.callback, interval = bot.interval_str)
#bsm.stop_socket(conn)


# client.get_asset_balance('BNB')
# client.get_asset_balance('USDC')

# buy_order = client.order_market_buy(symbol='BTCUSDC', quantity = 0.00104)
# sell_order = client.order_market_sell(symbol='BTCUSDC', quantity = 0.00104)
#bot.data.to_csv('C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/001_Bullish_ATR.csv')

#DEPLOYMENT
#py Documents\GitHub\AJbots\strategy_sandbox\001_Bullish_ATR_bar_strategy.py


