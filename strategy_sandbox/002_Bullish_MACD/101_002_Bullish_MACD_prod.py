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
from ta.trend import MACD
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

    def __init__(self, ticker, interval, hist_period_minutes, fast_ma, slow_ma, smooth=5, units=0):
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
        self.buy_amount_usd = 0
        self.sell_amount_usd = 0
        self.last_bar = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
        self.data = self.get_historical_data().assign(order_price=np.nan, order_qty=np.nan,commission=np.nan, bnb_topup=np.nan)
        self.bot_start_date = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).strftime("%d%b%Y_%H%M%S")
        self.bot_balance_filename = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/002_Bullish_MACD/101_002_Bullish_MACD_bal.csv'

        #*****************add strategy-specific attributes here******************
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.smooth = smooth
        #************************************************************************
        self.define_strategy()
        self.trades_data = pd.DataFrame(columns = ['raw_order_response','bnb_topup'])
        #dlja otslezhivanija neskoljkih strategij vnutri odnoj pary 
        if os.path.isfile(self.bot_balance_filename):
            self.balance = pd.read_csv('C:/Users/Administrator/Documents/GitHub/AJbots/production/002_Bullish_MACD/101_002_Bullish_MACD_bal.csv', index=False)
        else:
            self.balance = pd.DataFrame({'strategy_balance':[66.12],'initial_balance':[66.12]}).to_csv(self.bot_balance_filename, index=False)


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
        macd_obj = MACD(self.data.close, self.slow_ma,self.fast_ma,self.smooth)
        df['MACD'] = macd_obj.macd()
        df['MACD_hist'] = macd_obj.macd_diff()
        df['MACD_signal'] = macd_obj.macd_signal()
        df['position'] = np.where((df.MACD>0) & (np.sign(df.MACD_hist)>np.sign(df.MACD_hist.shift(1))), 1, np.nan)
        df['position'] = np.where((np.sign(df.MACD_hist)<np.sign(df.MACD_hist.shift(1))), 0, df.position)
        df['position']  = df.position.ffill().fillna(0)
        df['strategy_returns'] = df['position'].shift(1) * df['bench_returns']
        #***********************************************************************
        self.data = df.copy()

    def execute_trades(self):

        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:

            #Main order
                self.buy_order = client.order_market_buy(symbol=self.ticker, quantity=round_step_size(floor(float(client.get_asset_balance(asset='USDT')['free'])*0.95)/float(self.tick_data['close'].iloc[-1]), 0.001))

            #Technical fields
                self.bnb_price = float(client.get_symbol_ticker(symbol="BNBUSDT")['price'])
                self.data['order_price'].iloc[-1]= pd.json_normalize(self.buy_order['fills'])['price'].apply(pd.to_numeric).mean()
                self.data['order_qty'].iloc[-1] = pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum()
                self.data['commission'].iloc[-1] = pd.json_normalize(self.buy_order['fills'])['commission'].apply(pd.to_numeric).sum()*self.bnb_price
                self.buy_amount_usd = self.data['order_price'].iloc[-1] * self.data['order_qty'].iloc[-1]

                self.trades_data =self.trades_data.append(self.data.iloc[-1])
                self.trades_data['raw_order_response'].iloc[-1] = str(self.buy_order) 
                #prints
                file_name = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/002_Bullish_MACD/101_002_Bullish_MACD_tradelog_{}.csv'.format(self.bot_start_date)
                (self.balance.strategy_balance-self.data['commission'].iloc[-1]).to_csv(self.bot_balance_filename, index = False)
                self.trades_data.to_csv(file_name)
                self.report_trade(self.buy_order, "GOING LONG")
            self.position = 1
            
        elif self.data["position"].iloc[-1] == 0:
            if self.position == 1: 

            #Main order  
                self.sell_order = client.order_market_sell(symbol=self.ticker, quantity = pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum())

            #Technical fields
                self.bnb_price = float(client.get_symbol_ticker(symbol="BNBUSDT")['price'])
                self.data['order_price'].iloc[-1]= pd.json_normalize(self.sell_order['fills'])['price'].apply(pd.to_numeric).mean()
                self.data['order_qty'].iloc[-1] = pd.json_normalize(self.sell_order['fills'])['qty'].apply(pd.to_numeric).sum()
                self.data['commission'].iloc[-1] = pd.json_normalize(self.sell_order['fills'])['commission'].apply(pd.to_numeric).sum()*self.bnb_price
                self.sell_amount_usd = self.data['order_price'].iloc[-1] * self.data['order_qty'].iloc[-1]

                self.topup_bnb()
                self.trades_data =self.trades_data.append(self.data.iloc[-1])
                self.trades_data['raw_order_response'].iloc[-1] = str(self.sell_order) 
                #prints
                file_name = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/002_Bullish_MACD/101_002_Bullish_MACD_tradelog_{}.csv'.format(self.bot_start_date)
                self.balance.strategy_balance-self.data['commission'].iloc[-1]+(self.sell_amount_usd-self.buy_amount_usd).to_csv(self.bot_balance_filename, index = False)
                self.trades_data.to_csv(file_name)
                self.report_trade(self.sell_order, "GOING NEUTRAL")
            self.position = 0

        
    def report_trade(self, order, text):
        print("\n" + 100* "-")
        print(text)
        print("Filled quantity: {} - at price: {}".format(order['fills'][0]['qty'],order['fills'][0]['price']))
        print("Initial balance: {} | Actual balance {}".format(self.balance.initial_balance,self.balance.strategy_balance))
        print(100 * "-" + "\n")  


    def topup_bnb(self):
        ''' Top up BNB balance if it drops below minimum specified balance '''
        bnb_balance = float(client.get_asset_balance(asset='BNB')['free'])
        usd_balance = bnb_balance * self.bnb_price
        if usd_balance < 5:
            order = client.order_market_buy(symbol='BNBUSDT', quoteOrderQty=5)
            self.data['bnb_topup'].iloc[-1] = str(order)
            print('----------------BNB for commission is topped up!--------------')
        else: pass




#if __name__ == "__main__":
    
#bot = Trader(ticker = 'BTCUSDC',interval='15m', hist_period_minutes=100,  atr_period = 19, multiplicator=2.3, TP_atr=2.8, SL_atr=2.8) real one

bot = Trader(ticker = 'LTCUSDT',interval='5m', hist_period_minutes=100,  fast_ma = 15, slow_ma=10, smooth=3)

# init and start the WebSocket
bsm.start()
conn = bsm.start_kline_socket(symbol=bot.ticker, callback=bot.callback, interval = bot.interval_str)
#bsm.stop_socket(conn)

#DEPLOYMENT

# client.get_asset_balance('USDT')
# client.get_asset_balance('LTC')
# buy_order = client.order_market_buy(symbol='BTCUSDT', quantity = 0.001)
# sell_order = client.order_market_sell(symbol='LTCUSDT', quantity = 0.618)
#bot.trades_data.to_csv('C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/test.csv')

#py Documents\GitHub\AJbots\strategy_sandbox\001_Bullish_ATR_bar_strategy.py

