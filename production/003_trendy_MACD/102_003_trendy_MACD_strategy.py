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
import telegram_send as ts
import stored_functions as sf

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
# init
### API
with open("api_keys.json") as file:
    credentials = json.load(file)

binance_api_key = credentials['binance_api_key'] 
binance_api_secret = credentials['binance_api_secret']

ts_conf='telegram-send.conf'

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

bsm = ThreadedWebsocketManager()

class Trader():

    def __init__(self, ticker, interval, hist_period_minutes, fast_ma, slow_ma, smooth=5, pos_size_usd=0):
        self.ticker = ticker
        self.hist_period_minutes = hist_period_minutes
        self.tick_data = pd.DataFrame()
        self.interval = pd.to_timedelta(str(interval).replace('m','min'))
        self.interval_str = interval
        self.buy_order = dict()
        self.sell_order = dict()
        self.pos_size_usd = pos_size_usd
        self.position = 0
        self.bnb_price = 0
        self.buy_amount_usd = 0
        self.sell_amount_usd = 0
        self.last_bar = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
        self.last_print = self.last_bar
        self.data = self.get_historical_data().assign(order_price=np.nan, order_qty=np.nan,commission=np.nan, bnb_topup=np.nan)
        self.bot_start_date = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).strftime("%d%b%Y_%H%M%S")
        self.bot_balance_filename = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/003_trendy_MACD/102_003_trendy_MACD_bal.csv'
        self.file_name = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/003_trendy_MACD/102_003_trendy_MACD_tradelog_{}.csv'.format(self.bot_start_date)
        #*****************add strategy-specific attributes here******************
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.smooth = smooth
        #************************************************************************
        self.define_strategy()
        self.trades_data = pd.DataFrame(columns = ['raw_order_response','bnb_topup'])
        #dlja otslezhivanija neskoljkih strategij vnutri odnoj pary 
        if os.path.isfile(self.bot_balance_filename):
            self.balance = pd.read_csv(self.bot_balance_filename)
        else:
            self.balance = pd.DataFrame({'strategy_balance':[self.pos_size_usd],'initial_balance':[self.pos_size_usd]})
            self.balance.to_csv(self.bot_balance_filename, index = False)
        print('Bot is succesfully initiated!')


    def callback(self, df):
        global conn 
        recent_tick = pd.Timestamp.utcnow().tz_localize(None)
        
        # IF receives API error then it will restart the connection
        if df['e'] == 'error':
          bsm.stop_socket(conn)
          print('Socket stopped')
          ts.send(conf=ts_conf,messages=[df['m']])
          time.sleep(3)
          conn = bsm.start_kline_socket(symbol=self.ticker, callback=self.callback, interval = self.interval_str)


        if recent_tick - self.last_print > pd.to_timedelta('1h'):
            printed_price = "102_003 price: {} at: {}".format(round(float(df['k']['c']),2),recent_tick.strftime("%d %b %Y %H:%M"))
            ts.send(conf=ts_conf,messages=[str(printed_price)])
            print(printed_price)
            self.last_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
            
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
        df = client.get_historical_klines(symbol=self.ticker,interval = self.interval_str, start_str = timestamp)
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
        df['position'] = np.where((np.sign(df.MACD)<0) & (df.MACD>df.MACD_signal) & (df.MACD.shift(1)<df.MACD_signal.shift(1)),1,np.nan)
        df['position'] = np.where((np.sign(df.MACD)>0) & (df.MACD<df.MACD_signal) & (df.MACD.shift(1)>df.MACD_signal.shift(1)),0,df.position)
        df['position']  = df.position.ffill().fillna(0)
        df['strategy_returns'] = df['position'].shift(1) * df['bench_returns']
        #***********************************************************************
        self.data = df.copy()

    def execute_trades(self):

        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:

            #Main order
                self.buy_order = client.order_market_buy(symbol=self.ticker, quantity=round_step_size(float(self.pos_size_usd)/float(self.tick_data['close'].iloc[-1]), 0.01))

            #Technical fields
                self.bnb_price = float(client.get_symbol_ticker(symbol="BNBUSDT")['price'])
                self.data['order_price'].iloc[-1]= pd.json_normalize(self.buy_order['fills'])['price'].apply(pd.to_numeric).mean()
                self.data['order_qty'].iloc[-1] = pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum()
                self.data['commission'].iloc[-1] = pd.json_normalize(self.buy_order['fills'])['commission'].apply(pd.to_numeric).sum()*self.bnb_price
                self.buy_amount_usd = self.data['order_price'].iloc[-1] * self.data['order_qty'].iloc[-1]

                self.trades_data =self.trades_data.append(self.data.iloc[-1])
                self.trades_data['raw_order_response'].iloc[-1] = str(self.buy_order) 
                #prints
                self.balance = pd.read_csv(self.bot_balance_filename) 
                self.balance = pd.DataFrame({'strategy_balance':(self.balance.strategy_balance.iloc[0]-self.data['commission'].iloc[-1]),'initial_balance':[self.balance.initial_balance.iloc[0]]})
                self.balance.to_csv(self.bot_balance_filename, index = False)
                self.trades_data.to_csv(self.file_name)
                self.report_trade(self.buy_order, "___________102_003 GOING LONG___________")
            self.position = 1
            
        elif self.data["position"].iloc[-1] == 0:
            if self.position == 1: 

            #Main order  
                self.sell_order = client.order_market_sell(symbol=self.ticker, quantity = round_step_size(pd.json_normalize(self.buy_order['fills'])['qty'].apply(pd.to_numeric).sum(), 0.01))

            #Technical fields
                self.bnb_price = float(client.get_symbol_ticker(symbol="BNBUSDT")['price'])
                self.data['order_price'].iloc[-1]= pd.json_normalize(self.sell_order['fills'])['price'].apply(pd.to_numeric).mean()
                self.data['order_qty'].iloc[-1] = pd.json_normalize(self.sell_order['fills'])['qty'].apply(pd.to_numeric).sum()
                self.data['commission'].iloc[-1] = pd.json_normalize(self.sell_order['fills'])['commission'].apply(pd.to_numeric).sum()*self.bnb_price
                self.sell_amount_usd = self.data['order_price'].iloc[-1] * self.data['order_qty'].iloc[-1]

                self.trades_data =self.trades_data.append(self.data.iloc[-1])
                self.trades_data['raw_order_response'].iloc[-1] = str(self.sell_order) 
                #prints
                self.balance = pd.read_csv(self.bot_balance_filename)
                self.balance = pd.DataFrame({'strategy_balance':(self.balance.strategy_balance.iloc[0]-self.data['commission'].iloc[-1]+(self.sell_amount_usd-self.buy_amount_usd)),'initial_balance':[self.balance.initial_balance.iloc[0]]})
                self.balance.to_csv(self.bot_balance_filename, index = False)
                self.trades_data.to_csv(self.file_name)
                self.report_trade(self.sell_order, "__________102_003 GOING NEUTRAL_________")
                self.topup_bnb()
            self.position = 0

        
    def report_trade(self, order, text):
        print("\n" + 100* "-")
        print(text)
        print("Filled quantity: {} - at price: {}".format(order['fills'][0]['qty'],order['fills'][0]['price']))
        print("Initial balance: {} | Actual balance {}".format(self.balance.initial_balance.iloc[0],self.balance.strategy_balance.iloc[0]))
        print("Total P/L: {}".format(round((self.balance.strategy_balance.iloc[0]/self.balance.initial_balance.iloc[0])*100,2)))
        print(100 * "-" + "\n")  
        ts.send(conf=ts_conf,messages=[text+" price : "+str(round(float(order['fills'][0]['price']),2))+" P/L : "+str(round((self.balance.strategy_balance.iloc[0]/self.balance.initial_balance.iloc[0])*100,2))])

    def topup_bnb(self):
        ''' Top up BNB balance if it drops below minimum specified balance '''
        bnb_balance = float(client.get_asset_balance(asset='BNB')['free'])
        usd_balance = bnb_balance * self.bnb_price
        if usd_balance < 5:
            order = client.order_market_buy(symbol='BNBUSDT', quoteOrderQty=10)
            self.data['bnb_topup'].iloc[-1] = True
        else: pass
        print('BNB balance in USD: {}'.format(round(float(usd_balance),2)))
        ts.send(conf=ts_conf,messages=['BNB balance in USD: {}'.format(round(float(usd_balance),2))])



if __name__ == "__main__":

    bot = Trader(ticker = 'LINKUSDT',interval='30m', hist_period_minutes=100,  fast_ma = 15, slow_ma=40, smooth=7, pos_size_usd=50)

    # init and start the WebSocket
    bsm.start()
    conn = bsm.start_kline_socket(symbol=bot.ticker, callback=bot.callback, interval = bot.interval_str)

    time.sleep(5)
    print(bot.tick_data)
    print('API connection established!')

#bsm.stop_socket(conn)

#DEPLOYMENT

# client.get_asset_balance('USDT')
# client.get_asset_balance('LINK')
# buy_order = client.order_market_buy(symbol='BTCUSDT', quantity = 0.001)
# sell_order = client.order_market_sell(symbol='LINKUSDT', quantity = 3.93)
#bot.trades_data.to_csv('C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/test.csv')

#py Documents\GitHub\AJbots\production\003_trendy_MACD\102_003_trendy_MACD_strategy.py
#bot.tick_data

# bot.execute_trades()
# bot.position = 1
# bot.data["position"].iloc[-1] = 1
