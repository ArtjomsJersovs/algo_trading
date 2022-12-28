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
from ta.volatility import average_true_range
import ta.momentum as tm
import json
import telegram_send as ts
ts_conf='telegram-send.conf'
with open("api_keys.json") as file:
    keys = json.load(file)
### API
binance_api_key = keys['binance_api_key']
binance_api_secret = keys['binance_api_secret']

### CONSTANTS
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)

client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

bsm = un.BinanceWebSocketApiManager(exchange="binance.com-futures")

class Trader():

    def __init__(self, ticker, interval, hist_period_minutes, lookback_bars=100, ma_interval=20, tp_coef = 1, sl_coef = 1):
        self.ticker = ticker
        self.hist_period_minutes = hist_period_minutes
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
        self.breaks_local_high = 0
        self.breaks_local_low = 0
        self.RSI_neutral = False
        self.leverage = 1
        self.test_order = 'none'
        client.futures_change_leverage(symbol=ticker, leverage=self.leverage)
        #-------------------
        self.buy_amount_usd = 0
        self.sell_amount_usd = 0
        self.last_bar = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).floor("5min")
        self.last_print = self.last_bar
        self.last_order_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
        self.data = self.get_historical_data().assign(order_price=np.nan, order_qty=np.nan,trade_size_usd=np.nan)
        self.bot_start_date = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)).strftime("%d%b%Y_%H%M%S")
        #for server
        #self.bot_balance_filename = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/007_flat_false_breakout/103_007_flat_false_breakout_bal.csv'
        #self.file_name = 'C:/Users/Administrator/Documents/GitHub/AJbots/production/007_flat_false_breakout/103_007_flat_false_breakout_tradelog_{}.csv'.format(self.bot_start_date)
        #for local machine
        self.bot_balance_filename = 'C:/Users/artjoms.jersovs/github/AJbots/production/007_flat_false_breakout/103_007_flat_false_breakout_bal.csv'
        self.file_name = 'C:/Users/artjoms.jersovs/github/AJbots/production/007_flat_false_breakout/103_007_flat_false_breakout_tradelog_{}.csv'.format(self.bot_start_date)
        #*****************add strategy-specific attributes here******************
        self.lookback_bars = lookback_bars
        self.ma_interval = ma_interval
        self.tp_coef = tp_coef
        self.sl_coef = sl_coef
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

        recent_tick = pd.Timestamp.utcnow().tz_localize(None)
        
        if len(df)<5:
            pass
        else:
            if recent_tick - self.last_print > pd.to_timedelta('15m'):
                printed_price = "103_007 price: {} at: {}".format(round(float(df['kline']['close_price']),2),recent_tick.strftime("%d %b %Y %H:%M"))
                ts.send(conf=ts_conf,messages=[str(printed_price)])
                print(printed_price)
                self.last_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))

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
            
            if recent_tick - self.last_order_print > pd.to_timedelta('10s'):
                self.last_order_print = pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None))
                self.execute_trades()

    def get_historical_data(self):
        timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(minutes = self.hist_period_minutes*5+10)).strftime("%d %b %Y %H:%M:%S"))
        df = client.get_historical_klines(symbol=self.ticker,interval = self.interval_str, start_str = timestamp, klines_type=HistoricalKlinesType.FUTURES)
        df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open','high','low','close', 'volume']].apply(pd.to_numeric)
        df['skip_data'] = 1
        return df
  
    def resample_and_join(self):
        self.data = self.data.append(self.tick_data.resample(self.interval, label = 'right').last().ffill().apply(pd.to_numeric).iloc[:-1])  #ne bratj poslednij, chtoby v data byli toljko polnije bary
        self.data = self.data.iloc[-self.hist_period_minutes:] 
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.data.index[-1]

    def define_strategy(self): # "strategy-specific"
        df = self.data.apply(pd.to_numeric).copy()
        #df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))

        #******************** define your strategy here ************************
        df['bench_returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
        df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.close, window = self.ma_interval)
        df['RSI'] = tm.rsi(df.close, self.lookback_bars, fillna=True)
        #df['volume_ma'] = self.data['volume'].rolling(self.ma_interval).mean()
        df[f'max_price_{self.lookback_bars}'] = self.data['high'].rolling(self.lookback_bars).max()
        df[f'min_price_{self.lookback_bars}'] = self.data['low'].rolling(self.lookback_bars).min()
        
        df['breaks_local_high'] = (df[f'max_price_{self.lookback_bars}'].shift(1)<=df.close) & (df[f'max_price_{self.lookback_bars}']>df.close.shift(1))
        df['breaks_local_low'] = (df[f'min_price_{self.lookback_bars}'].shift(1)>=df.close) & (df[f'min_price_{self.lookback_bars}']<df.close.shift(1))
        df['RSI_neutral'] = (df.RSI >= 40) & (df.RSI <=60)
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
        self.report_trade(order, f"___________103_007 GOING {side}___________")
    
        
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
        if (self.data.breaks_local_low.iloc[-1] and self.data.RSI_neutral.iloc[-1] and self.position == 0) or (self.test_order=='long'):
            self.buy_order = client.futures_create_order(
                symbol=self.ticker,
                type='MARKET',
                side='BUY',
                newOrderRespType = 'RESULT',                    
                quantity=round_step_size((floor(float(client.futures_account_balance()[-1]['balance'])/10)*10)*self.leverage/float(self.data.close.iloc[-1]), 0.001)
            )
            self.position = 1
            self.trailing_stop = self.data.close.iloc[-1] - (self.data.ATR.iloc[-1]*self.sl_coef)
            self.start_price = self.data.close.iloc[-1]  
            self.handle_order_data(order=self.buy_order, side='LONG')

        elif (self.data.breaks_local_high.iloc[-1] and self.data.RSI_neutral.iloc[-1] and self.position == 0) or (self.test_order=='short'):
            self.sell_order = client.futures_create_order(
                symbol=self.ticker,
                type='MARKET',
                side='SELL',
                newOrderRespType = 'RESULT',                    
                quantity=round_step_size((floor(float(client.futures_account_balance()[-1]['balance'])/10)*10)*self.leverage/float(self.data.close.iloc[-1]), 0.001)
            )
            self.position = -1
            self.trailing_stop = self.data.close.iloc[-1] + (self.data.ATR.iloc[-1]*self.sl_coef)
            self.start_price = self.data.close.iloc[-1]  
            self.handle_order_data(order=self.sell_order, side='SHORT')         
            
        #CLOSE POSITIONS
        elif self.position == 1:
            if (self.data.close.iloc[-1]>(self.start_price + (self.data.ATR.iloc[-1]*self.tp_coef))) or (self.test_order=='long_tp'):
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
                self.handle_order_data(order=self.sell_order, side='NEUTRAL BY TP')  

            #close all if price dropped under stoploss
            elif (self.trailing_stop > self.data.close.iloc[-1]) or (self.test_order=='long_sl'):
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
            if (self.data.close.iloc[-1]<(self.start_price - (self.data.ATR.iloc[-1]*self.tp_coef))) or (self.test_order=='short_tp'):
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
                self.handle_order_data(order=self.buy_order, side='NEUTRAL BY TP') 

            #close all if price dropped under stoploss
            elif (self.trailing_stop < self.data.close.iloc[-1]) or (self.test_order=='short_sl'):
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

    bot = Trader(ticker = 'BTCBUSD', interval='5m', hist_period_minutes=40, lookback_bars=35, ma_interval=18, tp_coef = 6, sl_coef = 2)

    # init and start the WebSocket
    main_stream = bsm.create_stream(['kline_5m'], ['btcbusd'], process_stream_data=bot.callback,  output="UnicornFy")
    
    time.sleep(5)
    print(bot.tick_data)
    print('API connection established!')

    #bsm.stop_stream(main_stream)

#DEPLOYMENT

# client.get_asset_balance('USDT')
# client.get_asset_balance('LINK')
#bot.trades_data.to_csv('C:/Users/Administrator/Documents/GitHub/AJbots/strategy_sandbox/test.csv')
#py Documents\GitHub\AJbots\production\007_flat_false_breakout\103_007_flat_false_breakout_strategy.py
#bot.tick_data

# bot.trailing_stop
# bot.start_price - (bot.data.ATR.iloc[-1]*bot.tp_coef)
