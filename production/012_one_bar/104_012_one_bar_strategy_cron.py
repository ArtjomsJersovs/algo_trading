from math import nan, floor
import os
import json
import time
import requests
import datetime as dt
from urllib import request
from binance.helpers import date_to_milliseconds, round_step_size
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import HistoricalKlinesType
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range, BollingerBands
import stored_functions as sf
import telegram_send as ts
#ts_conf=r'C:\Users\Administrator\Documents\algo_trading\telegram-send.conf'
ts_conf=r'C:\Users\artjoms.jersovs\github\algo_trading\algo_trading\telegram-send.conf'

class Trader():

    def __init__(self, ticker, size, interval, hist_period_hours, leverage, ma_interval=15, bb_interval=30, body_size =0.8, sl_coef=1.25, vol_coef = 1):
        self.ticker = ticker
        self.hist_period_hours = hist_period_hours
        self.interval = pd.to_timedelta(str(interval).replace('m','min'))
        self.interval_str = interval
        self.buy_order = dict()
        self.sell_order = dict()
        self.result = 0
        #calculate real position size based on minimum rounding
        self.last_price = float(requests.get('https://www.binance.com/api/v3/ticker/price?symbol={}'.format(self.ticker)).json()['price'])
        self.size = round_step_size(max(float(size)/self.last_price,0.001), 0.001)
        self.size_usd = round(self.last_price*self.size,2)
        print(self.size_usd,'$ position size')
        self.test_order = 'none'
        self.leverage = leverage
        client.futures_change_leverage(symbol=ticker, leverage=leverage)
        self.run_date = pd.to_datetime(dt.datetime.now())  
        #-------------------
        #for server
        # self.bot_balance_filename = 'C:/Users/Administrator/Documents/algo_trading/production/012_one_bar/104_012_one_bar_bal.csv'
        # self.params_filename = 'C:/Users/Administrator/Documents/algo_trading/production/012_one_bar/btcbusd_1h_params.csv'
        # self.file_name = 'C:/Users/Administrator/Documents/algo_trading/production/012_one_bar/104_012_one_bar_tradelog_{}.csv'.format(self.self.run_date)
        #for local machine
        self.bot_balance_filename = 'C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/production/012_one_bar/104_012_btcbusd_1h_bal.csv'
        self.params_filename = 'C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/production/012_one_bar/104_012_btcbusd_1h_params.csv'
        self.file_name = 'C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/production/012_one_bar/104_012_btcbusd_1h_tradelog.csv'
        #*****************add strategy-specific attributes here******************
        self.ma_interval = ma_interval
        self.bb_interval = bb_interval
        self.body_size = body_size
        self.sl_coef = sl_coef
        self.vol_coef = vol_coef
        #************************************************************************
        self.trades_data = pd.DataFrame(columns = ['raw_order_response'])
        #Read all params and logs
        #balance tracking
        if os.path.isfile(self.bot_balance_filename):
            self.balance = pd.read_csv(self.bot_balance_filename)
        else:
            self.balance = pd.DataFrame({'strategy_balance':[self.size_usd],'initial_balance':[self.size_usd]})
            self.balance.to_csv(self.bot_balance_filename, index = False)
        print('balance file - success')
        print(self.balance)
        
        #Read main trade parameters
        if os.path.isfile(self.params_filename):
            self.params = pd.read_csv(self.params_filename)
        else:
            self.params = pd.DataFrame({'date':[self.run_date],'position':[0],'start_price':[0],'trailing_stop':[0]})
            self.params.to_csv(self.params_filename, index = False) 
        self.position = self.params.position.loc[0]
        self.trailing_stop = self.params.trailing_stop[0]
        self.start_price = self.params.start_price[0]
        print('parameters file - success')
        print(self.params)
        #************************************************************************
        #MAIN CALLBACK
        self.data = self.get_historical_data().assign(order_price=np.nan, order_qty=np.nan,trade_size_usd=np.nan)
        self.define_strategy()
        self.execute_trades()


    def get_historical_data(self):
        timestamp = date_to_milliseconds(pd.to_datetime(pd.Timestamp.utcnow().tz_localize(None)-pd.Timedelta(hours = self.hist_period_hours)).strftime("%d %b %Y %H:%M:%S"))
        df = client.get_historical_klines(symbol=self.ticker,interval = self.interval_str, start_str = timestamp, klines_type=HistoricalKlinesType.FUTURES)
        df = pd.DataFrame(df, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open','high','low','close', 'volume']].apply(pd.to_numeric)
        print('historical data - success')
        return df

    def define_strategy(self): 
        df = self.data.apply(pd.to_numeric).copy()
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
        print('calculate strategy - success')
        
    def handle_order_data(self, order, side='LONG'):
        
        self.data.loc[-1,('order_price')]= float(order['avgPrice'])
        self.data.loc[-1,('order_qty')]= float(order['executedQty'])
        self.data.loc[-1,('trade_size_usd')] = float(order['cumQuote'])        
        self.trades_data =self.trades_data.append(self.data.iloc[-1])
        self.trades_data.loc[-1,('raw_order_response')] = str(order)
        self.result = self.balance.initial_balance.iloc[-1] + float(client.futures_account_trades()[-1]['realizedPnl']) + (float(client.futures_account_trades()[-1]['commission'])*(-1))
        
        self.params.loc[0,('position')] = self.position
        self.params.loc[0,('trailing_stop')] = self.trailing_stop
        self.params.loc[0,('start_price')] = self.start_price
        #prints
        self.balance = self.balance.append({'strategy_balance':self.result,'initial_balance':self.balance.initial_balance.iloc[0]}, ignore_index=True)
        self.balance.to_csv(self.bot_balance_filename, index = False)
        self.params.to_csv(self.params_filename, index = False)
        self.trades_data.to_csv(self.file_name)
        self.report_trade(order, f"___________104_012 GOING {side}___________")
    
    def report_trade(self, order, text):
        print("\n" + 100* "-")
        print(text)
        print("Filled quantity: {} - at price: {}".format(self.data['order_qty'].iloc[-1],self.data['order_price'].iloc[-1]))
        print("Initial balance: {} | Actual balance {}".format(self.balance.initial_balance.iloc[-1],self.balance.strategy_balance.iloc[-1]))
        print("Total P/L: {}".format(round((self.balance.strategy_balance.iloc[-1]/self.balance.initial_balance.iloc[-1])*100,2)))
        print(100 * "-" + "\n")  
        ts.send(conf=ts_conf, messages=[text+" price : "+str(round(float(order['avgPrice']),2))+" P/L : "+str(round((self.balance.strategy_balance.iloc[-1]/self.balance.initial_balance.iloc[-1])*100,2))])

    def execute_trades(self):
        #OPEN POSITION
        if self.position == 0:
            if (self.data.higher_low.iloc[-1]==1 and self.data.higher_close.iloc[-1]==1 and self.data.body_size.iloc[-1]==1 and self.data.over_ma_volume.iloc[-1]==1 and (self.data.bb_bbl.iloc[-1]<self.data.close.iloc[-1]) and (self.data.bb_bbh.iloc[-1]>self.data.close.iloc[-1])) or (self.test_order=='long'):
                self.buy_order = client.futures_create_order(
                    symbol=self.ticker,
                    type='MARKET',
                    side='BUY',
                    newOrderRespType = 'RESULT',                    
                    quantity=self.size
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
                    quantity=self.size
                )
                self.position = -1
                self.trailing_stop = self.data.close.iloc[-1] + (self.data.ATR.iloc[-1]*self.sl_coef)
                self.start_price = self.data.close.iloc[-1]  
                self.handle_order_data(order=self.sell_order, side='SHORT')         
            
        #CLOSE POSITIONS
        elif self.position == 1:
            #close all if price dropped under stoploss
            if (self.trailing_stop > float(self.data.close.iloc[-1])) or (self.test_order=='long_sl'):
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
                self.params.loc[0,('date')] = self.run_date
                self.params.loc[0,('trailing_stop')] = self.trailing_stop
                self.params.to_csv(self.params_filename)
        
        elif self.position == -1:
            #close all if price dropped under stoploss
            if (self.trailing_stop < float(self.data.close.iloc[-1])) or (self.test_order=='short_sl'):
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
            elif np.sign(self.data.bench_returns.iloc[-1]) == -1 and (self.start_price>self.data.close.iloc[-1]):
                self.trailing_stop = self.data.close.iloc[-1] + (self.data.ATR.iloc[-1]*self.sl_coef)   
                self.params.loc[0,('date')] = self.run_date 
                self.params.loc[0,('trailing_stop')] = self.trailing_stop
                self.params.to_csv(self.params_filename)
        

if __name__ == "__main__":
    client = sf.setup_api_conn_binance_only()
    bot = Trader(ticker='BTCBUSD', size= 25, interval='1h', hist_period_hours=40, leverage=1, ma_interval=15, bb_interval=30, body_size =0.8, sl_coef=1.25, vol_coef = 1)
    print('done')

