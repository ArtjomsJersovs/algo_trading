from distutils.command.config import config
from ensurepip import version
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

### CONSTANTS
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)

# client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

# bsm = un.BinanceWebSocketApiManager(exchange="binance.com-futures")

class Trader():

    def __init__(self, ticker, interval, hist_period_hours, ma_interval=15, bb_interval=30, body_size =0.8, sl_coef=1.25, vol_coef = 1):
        self.ticker = ticker
        self.hist_period_hours = hist_period_hours
        self.tick_data = pd.DataFrame()
        self.interval = pd.to_timedelta(str(interval).replace('m','min'))
        self.interval_str = interval
        self.buy_order = dict()
        self.sell_order = dict()
        self.pos_size_usd = float(client.futures_account_balance()[-1]['balance'])
        #--strategy specific
        self.position = 0
        self.trailing_stop = 0
        self.start_price = 0
        self.leverage = 3
        self.test_order = 'none'
        client.futures_change_leverage(symbol=ticker, leverage=self.leverage)
        #-------------------
        self.buy_amount_usd = 0
        self.sell_amount_usd = 0
        self.last_bar = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).floor("1h")
        self.last_print = self.last_bar
        self.last_order_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
        self.data = self.get_historical_data().assign(order_price=np.nan, order_qty=np.nan,trade_size_usd=np.nan)
        self.bot_start_date = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).strftime("%d%b%Y_%H%M%S")
        #for server
        self.bot_balance_filename = 'C:/Users/Administrator/Documents/algo_trading/production/012_one_bar/104_012_one_bar_bal.csv'
        self.file_name = 'C:/Users/Administrator/Documents/algo_trading/production/012_one_bar/104_012_one_bar_tradelog_{}.csv'.format(self.bot_start_date)
        #for local machine
        #self.bot_balance_filename = 'C:/Users/artjoms.jersovs/github/AJbots/production/012_one_bar/104_012_one_bar_bal.csv'
        #self.file_name = 'C:/Users/artjoms.jersovs/github/AJbots/production/012_one_bar/104_012_one_bar_tradelog_{}.csv'.format(self.bot_start_date)
        #*****************add strategy-specific attributes here******************
        self.ma_interval = ma_interval
        self.bb_interval = bb_interval
        self.body_size = body_size
        self.sl_coef = sl_coef
        self.vol_coef = vol_coef
        #************************************************************************
        self.define_strategy()
        self.trades_data = pd.DataFrame(columns = ['raw_order_response'])
        #dlja otslezhivanija neskoljkih strategij vnutri odnoj pary 
        if os.path.isfile(self.bot_balance_filename):
            self.balance = pd.read_csv(self.bot_balance_filename)
        else:
            self.balance = pd.DataFrame({'strategy_balance':[self.pos_size_usd],'initial_balance':[self.pos_size_usd]})
            self.balance.to_csv(self.bot_balance_filename, index = False)
        print('Bot is succesfully initiated!')


    def callback(self, df):

        try:
            recent_tick = pd.Timestamp.utcnow().tz_localize(None)
            
            if len(df)<5:
                pass
            else:
                if recent_tick - self.last_print > pd.to_timedelta('1h'):
                    printed_price = "104_012 price: {} at: {}".format(round(float(df['kline']['close_price']),2),recent_tick.strftime("%d %b %Y %H:%M"))
                    ts.send(conf=ts_conf,messages=[str(printed_price)])
                    print(printed_price)
                    self.last_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).floor('1H')

                df = pd.DataFrame({"open":df['kline']['open_price'],
                                "high":df['kline']['high_price'],
                                "low":df['kline']['low_price'],
                                "close":df['kline']['close_price'],
                                "volume":df['kline']['base_volume']},
                                index = [pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))])
                self.tick_data = self.tick_data.append(df)

                if recent_tick - self.last_bar > self.interval:
                    self.resample_and_join()
                    self.define_strategy()
                    self.execute_trades()
                
                if (recent_tick - self.last_order_print > pd.to_timedelta('30s')) and self.position != 0:
                    self.last_order_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
                    self.execute_trades()
        except Exception as e: 
            ts.send(conf=ts_conf,messages=[str(e)])

    def get_historical_data(self):
        timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(hours = self.hist_period_hours)).strftime("%d %b %Y %H:%M:%S"))
        df = client.get_historical_klines(symbol=self.ticker,interval = self.interval_str, start_str = timestamp, klines_type=HistoricalKlinesType.FUTURES)
        df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open','high','low','close', 'volume']].apply(pd.to_numeric)
        df['skip_data'] = 1
        return df
  
    def resample_and_join(self):
        self.data = self.data.append(self.tick_data.resample(self.interval, label = 'right').last().ffill().apply(pd.to_numeric).iloc[:-1])  #ne bratj poslednij, chtoby v data byli toljko polnije bary
        self.data = self.data.iloc[-self.hist_period_hours:] 
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.data.index[-1]

    def define_strategy(self): # "strategy-specific"
        df = self.data.apply(pd.to_numeric).copy()
        #df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))

        #******************** define your strategy here ************************
        df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
        df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.close, window = self.ma_interval)
        indicator_bb = BollingerBands(close=df["close"], window=self.bb_interval, window_dev=2)
        df['close_ma'] = df['close'].rolling(self.ma_interval).mean()
        df['bb_bbm'] = indicator_bb.bollinger_mavg()
        df['bb_bbh'] = indicator_bb.bollinger_hband()
        df['bb_bbl'] = indicator_bb.bollinger_lband()
        df['volume_ma'] = df['volume'].rolling(self.ma_interval).mean()
        #strategy specific
        df['higher_low'] = np.where(df.low>df.low.shift(1),1,0)
        df['lower_high'] = np.where(df.high<df.high.shift(1),1,0)
        df['higher_close'] = np.where(df.close>df.close.shift(1),1,0)
        df['lower_close'] = np.where(df.close<df.close.shift(1),1,0)
        df['over_ma_volume'] = np.where(df.volume>(df.volume_ma * self.vol_coef),1,0)
        
        df['rng'] = abs(df.high-df.low)
        df['rng_body'] = abs(df.open-df.close)
        df['body_size'] = np.where(df.rng_body.div(df.rng)>self.body_size,1,0)
        #df['strategy_returns'] = df['position'].shift(1) * df['bench_returns']
        #***********************************************************************
        self.data = df.copy()
        
    def handle_order_data(self, order, side='LONG'):
        self.data['order_price'].iloc[-1]= float(order['avgPrice'])
        self.data['order_qty'].iloc[-1] = float(order['executedQty'])
        self.data['trade_size_usd'].iloc[-1] = float(order['cumQuote'])
        self.trades_data =self.trades_data.append(self.data.iloc[-1])
        self.trades_data['raw_order_response'].iloc[-1] = str(order) 
        #prints
        self.balance = pd.read_csv(self.bot_balance_filename) 
        self.balance = self.balance.append({'strategy_balance':float(client.futures_account_balance()[-1]['balance']),'initial_balance':self.balance.initial_balance.iloc[0]}, ignore_index=True)
        self.balance.to_csv(self.bot_balance_filename, index = False)
        self.trades_data.to_csv(self.file_name)
        self.report_trade(order, f"___________104_012 GOING {side}___________")
    
        
    def report_trade(self, order, text):
        print("\n" + 100* "-")
        print(text)
        print("Filled quantity: {} - at price: {}".format(self.data['order_qty'].iloc[-1],self.data['order_price'].iloc[-1]))
        print("Initial balance: {} | Actual balance {}".format(self.balance.initial_balance.iloc[-1],float(client.futures_account_balance()[-1]['balance'])))
        print("Total P/L: {}".format(round((self.balance.strategy_balance.iloc[-1]/self.balance.initial_balance.iloc[-1])*100,2)))
        print(100 * "-" + "\n")  
        ts.send(conf=ts_conf, messages=[text+" price : "+str(round(float(order['avgPrice']),2))+" P/L : "+str(round((self.balance.strategy_balance.iloc[-1]/self.balance.initial_balance.iloc[-1])*100,2))])

    
#strategy is defined, need to implement execute trades
    def execute_trades(self):
        #OPEN POSITION
        if self.position == 0:
            if (self.data.higher_low.iloc[-1]==1 and self.data.higher_close.iloc[-1]==1 and self.data.body_size.iloc[-1]==1 and self.data.over_ma_volume.iloc[-1]==1 and (self.data.bb_bbl.iloc[-1]<self.data.close.iloc[-1]) and (self.data.bb_bbh.iloc[-1]>self.data.close.iloc[-1])) or (self.test_order=='long'):
                self.buy_order = client.futures_create_order(
                    symbol=self.ticker,
                    type='MARKET',
                    side='BUY',
                    newOrderRespType = 'RESULT',                    
                    quantity=round_step_size((floor(float(client.futures_account_balance()[-1]['balance'])/10-1)*10)*self.leverage/float(self.data.close.iloc[-1]), 0.001)
                )
                self.position = 1
                self.trailing_stop = self.data.close.iloc[-1] - (self.data.ATR.iloc[-1]*self.sl_coef)
                self.start_price = self.data.close.iloc[-1]  
                self.handle_order_data(order=self.buy_order, side='LONG')

            elif (self.data.lower_high.iloc[-1]==1 and self.data.lower_close.iloc[-1]==1 and self.data.body_size.iloc[-1]==1 and self.data.over_ma_volume.iloc[-1]==1 and (self.data.bb_bbh.iloc[-1]>self.data.close.iloc[-1]) and (self.data.bb_bbl.iloc[-1]<self.data.close.iloc[-1])) or (self.test_order=='short'):
                self.sell_order = client.futures_create_order(
                    symbol=self.ticker,
                    type='MARKET',
                    side='SELL',
                    newOrderRespType = 'RESULT',                    
                    quantity=round_step_size((floor(float(client.futures_account_balance()[-1]['balance'])/10-1)*10)*self.leverage/float(self.data.close.iloc[-1]), 0.001)
                )
                self.position = -1
                self.trailing_stop = self.data.close.iloc[-1] + (self.data.ATR.iloc[-1]*self.sl_coef)
                self.start_price = self.data.close.iloc[-1]  
                self.handle_order_data(order=self.sell_order, side='SHORT')         
            
        #CLOSE POSITIONS
        elif self.position == 1:
            #close all if price dropped under stoploss
            if (self.trailing_stop > float(self.tick_data.close.iloc[-1])) or (self.test_order=='long_sl'):
                self.sell_order = client.futures_create_order(
                    symbol=self.ticker,
                    type='MARKET',
                    side='SELL',
                    newOrderRespType = 'RESULT',
                    quantity=float(client.futures_get_all_orders(symbol=self.ticker)[-1]['origQty'])
                )
                self.position = 0
                self.trailing_stop = 0
                self.start_price = 0
                self.handle_order_data(order=self.sell_order, side='NEUTRAL BY SL')

            #if no actions with position, then pull up stoploss after price climbed over pinbar high + ATR
            elif np.sign(self.data.bench_returns.iloc[-1]) == 1 and (self.start_price<self.data.close.iloc[-1]):
                self.trailing_stop = self.data.close.iloc[-1] - (self.data.ATR.iloc[-1]*self.sl_coef)   
        
        elif self.position == -1:
            #close all if price dropped under stoploss
            if (self.trailing_stop < float(self.tick_data.close.iloc[-1])) or (self.test_order=='short_sl'):
                self.buy_order = client.futures_create_order(
                    symbol=self.ticker,
                    type='MARKET',
                    side='BUY',
                    newOrderRespType = 'RESULT',
                    quantity=float(client.futures_get_all_orders(symbol=self.ticker)[-1]['origQty'])
                )
                self.position = 0
                self.trailing_stop = 0
                self.start_price = 0
                self.handle_order_data(order=self.buy_order, side='NEUTRAL BY SL')

            #if no actions with position, then pull up stoploss after price climbed over pinbar high + ATR
            elif (np.sign(self.data.bench_returns.iloc[-1]) == -1 and (self.start_price>self.data.close.iloc[-1])) or (self.test_order=='trail'):
                self.trailing_stop = self.data.close.iloc[-1] + (self.data.ATR.iloc[-1]*self.sl_coef)   
        

if __name__ == "__main__":

    bot = Trader(ticker = 'BTCBUSD', interval='1h', hist_period_hours=40, ma_interval=15, bb_interval=30, body_size =0.8, sl_coef=1.25, vol_coef = 1)

    # init and start the WebSocket
    main_stream = bsm.create_stream(['kline_1h'], ['btcbusd'], process_stream_data=bot.callback,  output="UnicornFy")
    
    time.sleep(5)
    print(bot.tick_data)
    print('API connection established!')

    #bsm.stop_stream(main_stream)
#DEPLOYMENT

# client.get_asset_balance('USDT')
# client.get_asset_balance('LINK')
#bot.trades_data.to_csv('C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/test.csv')
#py C:\Users\Administrator\Documents\algo_trading\production\012_one_bar\104_012_one_bar_strategy.py
#bot.tick_data

# bot.test_order='long'
# bot.execute_trades()

