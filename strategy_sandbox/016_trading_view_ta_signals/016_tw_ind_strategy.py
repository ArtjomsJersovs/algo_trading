# Decription - strategy for flat market for bounce back from local extremums by using trailing stop on ATR

import os
import time
import datetime as dt
import time
from binance.helpers import date_to_milliseconds
import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range, BollingerBands
import ta.momentum as tm
import stored_functions as sf
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import pyplot
import itertools
plt.style.use("ggplot")


pd.set_option('display.max_rows', 100)
#INIT API
client, bsm = sf.setup_api_conn_binance()

class IterativeBase():
    def __init__(self, symbol, amount, tf = '5m'):

        #DEFINE MAIN PARAMETERS
        self.symbol = symbol
        self.initial_balance = amount
        self.current_balance = amount 
        self.tf_orig = str(tf)
        self.tf = pd.to_timedelta(str(tf).replace('m','min')) 
        self.units = 0
        self.trades = 0
        self.costs = 0 
        self.position = 0
        self.cpv = 0
        self.results = None 
        self.df = self.preprocess_data()


    def __repr__(self):
        rep = "IterativeBase(symbol = {}, start = {}, end = {}"
        return rep.format(self.symbol)

    def preprocess_data(self):
        filename = r'C:\Users\artjoms.jersovs\github\algo_trading\algo_trading\strategy_sandbox\016_trading_view_ta_signals\tw_data_clean.xlsx'
        raw = pd.read_excel(filename)
        raw['timestamp'] = pd.to_datetime(raw['timestamp'])
        raw.set_index('timestamp', inplace=True)
        df = raw.dropna()#.apply(pd.to_numeric)
        #df.dropna(inplace=True)
        self.data = df 
        return print(self.data.head(10))

    def plot_data(self, cols = None):
        if cols is None:
            cols = 'close'
        #self.data[cols].plot(figsize=(12,8), title = self.symbol)
        self.data[['cumreturns','cumstrategy']].plot(title = 'Cum returns comparison', figsize=(12, 8),alpha=0.5)
        return plt.show()

    def get_values(self, bar):
        date = str(self.data.index[bar].date())
        price = self.data.close.iloc[bar]
        return date, price

    def print_current_balance(self, bar):
        date, price = self.get_values(bar)
        print('current balance: {} | Date: {}'.format(round(self.current_balance,5),date))

    def buy_instrument(self, bar, units = None, amount = None, commission = 0.00075):

        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = float(amount / price)
        self.units += abs(units)
        self.trades += 1
        costs = (units * price) * commission
        self.costs += costs
        self.current_balance -= abs(units) * price # reduce cash balance by "purchase price"
        #print("{} |  Buying {} for {}, commission paid: {}".format(date, round(units,5), round(price, 5), round(costs,2)))

    def sell_instrument(self, bar, units = None, amount = None, percent = 1, commission = 0.00075):

        date, price = self.get_values(bar)

        if units is None and amount is None:
            units = self.units * percent
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = float(amount / price)
        self.units -= units
        self.trades += 1
        costs = (abs(units) * price) * commission
        self.costs +=costs
        self.current_balance += units * price # reduce cash balance by "purchase price"
       # print("{} |  Selling {} for {}, commission paid: {}".format(date, round(units,5), round(price, 5), round(costs,2)))

    def current_position_value(self, bar):
        date, price = self.get_values(bar)
        cpv = self.units * price 
        print("{} | Current position value = {}".format(date,round(cpv,5)))

    def print_current_net_asset_value(self, bar):
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} | Net asset value = {}".format(date, round(nav, 5)))

    def close_all(self, bar, commission = 0.001):
        date, price = self.get_values(bar)
        # print(75 * "-")
        # print("FINAL CLOSE OF ALL POSITIONS")
        #print("Closing position of {} for {}".format(round(self.units,5), price))

        self.trades += 1
        costs = (self.units * price) * commission
        self.costs += costs
        self.current_balance += self.units * price
        perf_abs = self.current_balance - self.initial_balance -  self.costs
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        perf_w_comm = (self.current_balance - self.initial_balance - self.costs) / self.initial_balance * 100
        self.units = 0

        self.data['strategy'] = self.data['position'].shift(1) * self.data['returns']
        self.data['cumstrategy'] = self.data['strategy'].cumsum().apply(np.exp)
        self.data['bench_cummax'] = self.data.cumreturns.cummax() 
        self.data['bench_drawdown'] = (self.data['bench_cummax'] - self.data['cumreturns']) / self.data['bench_cummax']
        self.data['strategy_cummax'] = self.data.cumstrategy.cummax() 
        self.data['strategy_drawdown'] = (self.data['strategy_cummax'] - self.data['cumstrategy']) / self.data['strategy_cummax']

        month = pd.to_timedelta(str('30d'))
        bench_monthly_return = self.data.returns.mean() * (month/self.tf) #returns in % monthly
        bench_monthly_var = self.data.returns.std() * np.sqrt(month/self.tf) #risk in % monthly #make sense only in normally distributed returns as std and mean is used
        bench_max_drawdown = self.data.bench_drawdown.max()

        strategy_monthly_return = self.data.strategy.mean() * (month/self.tf)
        strategy_monthly_var = self.data.strategy.std() * np.sqrt(month/self.tf) 
        strategy_max_drawdown = self.data.strategy_drawdown.max()

        print(75 * "-")
        print('{}% - Benchmark monthly return'.format(round(bench_monthly_return*100,2)))
        print('{}% - Strategy monthly return'.format(round(strategy_monthly_return*100,2)))
        print(75 * "-")
        print('{}% - Benchmark monthly VaR'.format(round(bench_monthly_var*100,2)))
        print('{}% - Strategy monthly VaR'.format(round(strategy_monthly_var*100,2)))
        print(75 * "-")
        print('{}% - Benchmark Maximum drawdown on {}'.format(round(bench_max_drawdown*100,2),self.data.bench_drawdown.idxmax()))
        print('{}% - Strategy Maximum drawdown on {}'.format(round(strategy_max_drawdown*100,2),self.data.strategy_drawdown.idxmax()))
        print(75 * "-")

        print("Net Performance: {}%".format(round(perf,2)))
        print("Performance with commissions in %: {} | absolute: {}$".format(round(perf_w_comm,2),round(perf_abs,2)))
        print("Trades executed: {} | commisions paid: {}".format(self.trades,round(self.costs,2)))
        print("Initial balance was : {}".format(round(self.initial_balance,2)))
        print('Current balance is {}'.format(self.current_balance - self.costs))
        print('Accuracy is : {}'.format(round(len(self.data.loc[self.data['pos_result']==1,['pos_result']])/len(self.data.loc[self.data['pos_result']!=0,['pos_result']]),2)))

        print(75 * "-")


class IterativeBacktest(IterativeBase):

    # helper method
    def close_long(self, bar, size):
        if size == 'all':
            self.sell_instrument(bar, units = self.units)
        elif size == 'half':
            self.sell_instrument(bar, units = (self.units/2))
            
    def close_short(self, bar, size):
        if size == 'all':
            self.buy_instrument(bar, units = self.units)
        elif size == 'half':
            self.buy_instrument(bar, units = (self.units/2))
            
    def go_long(self, bar, units = None, amount = None):
        if self.position == -1:
            self.close_short(bar, size = 'all') #if short , goneutral first
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == 'all':
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount)

    def go_short(self, bar, units = None, amount = None):
        if self.position == 1:
            self.close_long(bar, size = 'all') #if short , goneutral first
        if units:
            self.sell_instrument(bar, amount = amount)
        elif amount:
            if amount == 'all':
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount)

              
    def calculate_tw_strategy(self):
        # reset all parameters
        self.position = 0
        self.trades = 0
        self.costs = 0
        self.current_balance = self.initial_balance
        self.trailing_stop = 0
        self.start_price = 0
        self.pos_result = 0
        
        self.data['position'] = 0
        self.data['trailing_stop'] = 0
        self.data['pos_result'] = 0
        self.data['start_price'] = 0
        #strategy specific
        self.data['ATR'] = average_true_range(high = self.data.high, low = self.data.low, close = self.data.close, window = 15)
        self.data['returns'] = np.log(self.data.close.astype(float).div(self.data.close.astype(float).shift(1)))
        self.data['cumreturns'] = self.data['returns'].cumsum().apply(np.exp)
        #self.data.dropna(inplace = True)
        
        for bar in tqdm(range(len(self.data)-1)): # all bars except of last one
        #for bar in range(len(self.data)-1):
            self.pos_result = 0
        #CLOSE POSITIONS
            if self.position == 1:
                # if self.data.close.iloc[bar] > self.start_price and self.data.close_ma.iloc[bar-1]>self.data.bb_bbm.iloc[bar-1] and self.data.close_ma.iloc[bar]<self.data.bb_bbm.iloc[bar]:
                #     self.close_long(bar, size='all')
                #     self.pos_result = 1
                #     self.position = 0
                #     self.trailing_stop = 0
                #     self.start_price = 0
                
                if self.trailing_stop > self.data.close.iloc[bar]:
                    self.close_long(bar, size='all')
                    self.pos_result = np.where(self.data.close.iloc[bar]>self.start_price,1,-1)
                    self.position = 0
                    self.trailing_stop = 0
                    self.start_price = 0
                    
                elif np.sign(self.data.returns.iloc[bar]) == 1 and (self.start_price<self.data.close.iloc[bar]):
                    self.trailing_stop = self.data.close.iloc[bar] - (self.data.ATR.iloc[bar]*1)
                    
                    
            elif self.position == -1:
                # if self.data.close.iloc[bar] < self.start_price and self.data.close_ma.iloc[bar-1]<self.data.bb_bbm.iloc[bar-1] and self.data.close_ma.iloc[bar]>self.data.bb_bbm.iloc[bar]:
                #     self.close_short(bar, size='all')
                #     self.pos_result = 1
                #     self.position = 0
                #     self.trailing_stop = 0
                #     self.start_price = 0
                    
                if self.trailing_stop < self.data.close.iloc[bar]:
                    self.close_short(bar, size='all')
                    self.pos_result = np.where(self.data.close.iloc[bar]<self.start_price,1,-1)
                    self.position = 0
                    self.trailing_stop = 0
                    self.start_price = 0
                
                elif np.sign(self.data.returns.iloc[bar]) == -1 and (self.start_price>self.data.close.iloc[bar]):
                    self.trailing_stop = self.data.close.iloc[bar] + (self.data.ATR.iloc[bar]*1)


        #OPEN POSITIONS
            if self.position == 0:
                if  self.data.rec_sum_buy.iloc[bar]<=2 and self.data.rec_sum_sell.iloc[bar]>=12:
                    self.go_long(bar,amount = 'all')
                    self.position = 1
                    self.trailing_stop = self.data.close.iloc[bar] - (self.data.ATR.iloc[bar])
                    self.start_price = self.data.close.iloc[bar]  
                    self.data['start_price'].iloc[bar] = self.start_price
                               
                elif self.data.rec_sum_sell.iloc[bar]<=2 and self.data.rec_sum_buy.iloc[bar]>=12:
                    self.go_short(bar,amount = 'all')
                    self.position = -1
                    self.trailing_stop = self.data.close.iloc[bar] + (self.data.ATR.iloc[bar])
                    self.start_price = self.data.close.iloc[bar]   
                    self.data['start_price'].iloc[bar] = self.start_price
                    

            self.data['position'].iloc[bar] = self.position
            self.data['trailing_stop'].iloc[bar] = self.trailing_stop
            self.data['pos_result'].iloc[bar] = self.pos_result
        self.close_all(bar+1) # close pos at the last bar       
        

bc = IterativeBacktest("BTCBUSD",amount = 1000)

bc.calculate_tw_strategy()
bc.plot_data()

# sf.excel_export(bc.data.tail(10000))


#Results - standalone TW indicators works in opposite. If fewer indicators points on BUY - then its better to BUY.
# Most important featuers rec_sum_buy / sell / neutral doesnt give the edge over market.

