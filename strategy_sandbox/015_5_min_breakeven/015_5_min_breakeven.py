import os
import time
import datetime as dt
import time
import math
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
import xgboost as xgb
from ast import Index
plt.style.use("ggplot")


pd.set_option('display.max_rows', 100)
#INIT API
client, bsm = sf.setup_api_conn_binance()

sf.get_all_binance('BTCBUSD', '5m', save=True)
# filename = str(os.getcwd())+'\\strategy_sandbox\\datasets\\BTCBUSD-5m-data.csv' 
# raw = pd.read_csv(filename)
# raw['timestamp'] = pd.to_datetime(raw['timestamp'])
# raw.set_index('timestamp', inplace=True)
# raw['ATR'] = average_true_range(high = raw.high, low = raw.low, close = raw.close, window = 15)
# raw['month'] = pd.DatetimeIndex(raw.index).month
# raw['atr_to_price_ratio'] = raw['ATR']/raw['close']
# df = raw.copy()

# sf.get_stored_data_close('BTCBUSD','1h',"2020-01-01","2022-11-28")

# df.groupby(['month']).ATR.describe()
#         count       mean        std        min        25%        50%        75%         max
# month
# 1      8928.0  35.492061  25.027949   3.162617  15.008896  30.582337  48.265758  175.733918
# 2      8064.0  43.750409  22.597319   7.833074  27.508615  37.768233  55.390842  148.333925
# 3      6023.0  63.794312  39.244365  11.033762  32.864143  55.019987  87.272289  275.371851
# 10     8928.0  32.579469  15.352405   0.000000  21.889042  29.714862  40.602386  118.626729
# 11     8640.0  43.908058  37.051062   9.142804  22.961354  31.613394  48.803915  343.508883
# 12     8928.0  17.930087   9.832474   4.119455  11.587079  15.913845  21.693625  107.035028
# >>>
# df.groupby(['month']).atr_to_price_ratio.describe()
#         count      mean       std       min       25%       50%       75%       max
# month
# 1      8928.0  0.001690  0.001121  0.000187  0.000855  0.001466  0.002206  0.008411
# 2      8064.0  0.001865  0.000932  0.000359  0.001190  0.001619  0.002364  0.005939
# 3      6027.0  0.002641  0.001526  0.000494  0.001474  0.002341  0.003493  0.010673
# 10     8928.0  0.001654  0.000768  0.000000  0.001125  0.001512  0.002055  0.006473
# 11     8640.0  0.002514  0.002139  0.000552  0.001259  0.001809  0.002873  0.018894
# 12     8928.0  0.001055  0.000566  0.000245  0.000687  0.000939  0.001277  0.005976
# >>>
#commission for trade is 0.0015
#mean 5 min bar atr is 0.001857
# df.atr_to_price_ratio.describe()
#Min loss will be 0.00335
#with 50% accuracy 
#profit should be 0.00485 with loss 0.0019
#roughly profit ratio should be 2.6 for breakeven

#profit ratio 3:1 will earn of average 0.085% per win trade, 0.0425% per every trade
# ((3 * 0.0019) - 0.00485)*100
#in order to earn 15% monthly its needed to execute 12 trades per day with 3:1 profit ratio and 50% accuracy.
# 15/0.0425/30


class IterativeBase():
    def __init__(self, symbol, start, end, amount, tf = '15m'):

        #DEFINE MAIN PARAMETERS
        self.symbol = symbol
        self.start = start
        self.end = end
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
        return rep.format(self.symbol, self.start, self.end)

    def preprocess_data(self):
        filename = str(os.getcwd())+'\\strategy_sandbox\\datasets\\%s-%s-data.csv' % (self.symbol, self.tf_orig)
        if os.path.isfile(filename):
            raw = pd.read_csv(filename)
            raw['timestamp'] = pd.to_datetime(raw['timestamp'])
            raw.set_index('timestamp', inplace=True)
            raw = raw.resample(self.tf, label = 'right').last().ffill().apply(pd.to_numeric)
            print('All caught up! Existing dataset is used.')
        else:
            raw = sf.get_all_binance(self.symbol, self.tf_orig, save=True)
            
        
        df = raw.loc[str(self.start):str(self.end)].copy()
        df = df.dropna().apply(pd.to_numeric)
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

    def buy_instrument(self, bar, units = None, amount = None, trail_stop=False, trail_stop_price = 0, commission = 0.00075):

        date, price = self.get_values(bar)
        if trail_stop == True: 
            price = trail_stop_price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = float(amount / price)
        self.units += abs(units)
        self.trades += 1
        costs = (units * price) * commission
        self.costs += costs
        self.current_balance -= abs(units) * price # reduce cash balance by "purchase price"
        #print("{} |  Buying {} for {}, commission paid: {}".format(date, round(units,5), round(price, 5), round(costs,2)))

    def sell_instrument(self, bar, units = None, amount = None,trail_stop=False, trail_stop_price = 0, percent = 1, commission = 0.00075):

        date, price = self.get_values(bar)
        if trail_stop == True: 
            price = trail_stop_price
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
            self.sell_instrument(bar, units = self.units, trail_stop=True,  trail_stop_price=self.trailing_stop)
        elif size == 'half':
            self.sell_instrument(bar, units = (self.units/2), trail_stop=True,  trail_stop_price=self.trailing_stop)
            
    def close_short(self, bar, size):
        if size == 'all':
            self.buy_instrument(bar, units = self.units, trail_stop=True,  trail_stop_price=self.trailing_stop)
        elif size == 'half':
            self.buy_instrument(bar, units = (self.units/2), trail_stop=True,  trail_stop_price=self.trailing_stop)
            
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

              
    def calculate_test_strategy(self, ma_interval=10, bb_interval=20, body_size =0.8, sl_coef=0.0019, tp_coef = 0.005):
        # reset all parameters
        self.position = 0
        self.trades = 0
        self.costs = 0
        self.current_balance = self.initial_balance
        self.trailing_stop = 0
        self.start_price = 0
        self.pos_result = 0
        self.rng_vs_price_start = 0
        
        self.data['ATR'] = average_true_range(high = self.data.high, low = self.data.low, close = self.data.close, window = ma_interval)
        indicator_bb = BollingerBands(close=self.data["close"], window=bb_interval, window_dev=2)
        self.data['position'] = 0
        self.data['trailing_stop'] = 0
        self.data['pos_result'] = 0
        self.data['start_price'] = 0
        self.data['rng_vs_price_start'] = 0
        
        #strategy specific
        self.data['volume_ma'] = self.data['volume'].rolling(ma_interval).mean()
        self.data['higher_low'] = np.where(self.data.low>self.data.low.shift(1),1,0)
        self.data['lower_high'] = np.where(self.data.high<self.data.high.shift(1),1,0)
        self.data['higher_close'] = np.where(self.data.close>self.data.close.shift(1),1,0)
        self.data['lower_close'] = np.where(self.data.close<self.data.close.shift(1),1,0)
        self.data['over_ma_volume'] = np.where(self.data.volume>self.data.volume_ma,1,0)
        self.data['rng'] = abs(self.data.high-self.data.low)
        self.data['rng_body'] = abs(self.data.open-self.data.close)
        self.data['body_size'] = np.where(self.data.rng_body.div(self.data.rng)>0.8,1,0)
        self.data['soldier_nr_pos'] = 0
        self.data['soldier_nr_neg'] = 0
        self.data['rng_vs_price'] = np.where((np.ceil(round(self.data.rng/self.data.close,5)/0.001)*0.001)>0.006,0.006,(np.ceil(round(self.data.rng/self.data.close,5)/0.001)*0.001))
        
        self.data['returns'] = np.log(self.data.close.astype(float).div(self.data.close.astype(float).shift(1)))
        self.data['cumreturns'] = self.data['returns'].cumsum().apply(np.exp)
        #self.data.dropna(inplace = True)
        
        for bar in tqdm(range(len(self.data)-1)): # all bars except of last one
        #for bar in range(len(self.data)-1):
            self.pos_result = 0
            if self.data.soldier_nr_pos.iloc[bar-1]==0 and self.data.higher_low.iloc[bar]==1 and self.data.higher_close.iloc[bar]==1 and self.data.volume.iloc[bar]>self.data.volume.iloc[bar-1]:
                self.data.soldier_nr_pos.iloc[bar] = 1
            elif self.data.soldier_nr_pos.iloc[bar-1] == 1 and self.data.higher_low.iloc[bar]==1 and self.data.higher_close.iloc[bar]==1 and self.data.volume.iloc[bar]>self.data.volume.iloc[bar-1]:
                self.data.soldier_nr_pos.iloc[bar] = self.data.soldier_nr_pos.iloc[bar-1]+1
            elif self.data.soldier_nr_pos.iloc[bar-1] == 2 and self.data.higher_low.iloc[bar]==1 and self.data.higher_close.iloc[bar]==1:
                self.data.soldier_nr_pos.iloc[bar] = self.data.soldier_nr_pos.iloc[bar-1]+1
            else:
                self.data.soldier_nr_pos.iloc[bar] = 0     
                
            if self.data.soldier_nr_neg.iloc[bar-1]==0 and self.data.lower_high.iloc[bar]==1 and self.data.lower_close.iloc[bar]==1 and self.data.volume.iloc[bar]>self.data.volume.iloc[bar-1]:
                self.data.soldier_nr_neg.iloc[bar] = 1
            elif self.data.soldier_nr_neg.iloc[bar-1] == 1 and self.data.lower_high.iloc[bar]==1 and self.data.lower_close.iloc[bar]==1 and self.data.volume.iloc[bar]>self.data.volume.iloc[bar-1]:
                self.data.soldier_nr_neg.iloc[bar] = self.data.soldier_nr_neg.iloc[bar-1]+1
            elif self.data.soldier_nr_neg.iloc[bar-1] == 2 and self.data.lower_high.iloc[bar]==1 and self.data.lower_close.iloc[bar]==1:
                self.data.soldier_nr_neg.iloc[bar] = self.data.soldier_nr_neg.iloc[bar-1]+1
            else:
                self.data.soldier_nr_neg.iloc[bar] = 0                                
        #CLOSE POSITIONS
            if self.position == 1:
                #stoploss
                if self.start_price*(1-sl_coef)>=self.data.close.iloc[bar]:
                    self.close_long(bar, size='all')
                    self.pos_result = -1
                    self.position = 0
                    self.trailing_stop = 0
                    self.start_price = 0
                elif self.start_price*(1+tp_coef)<=self.data.close.iloc[bar]:
                    self.close_long(bar, size='all')
                    self.pos_result = 1
                    self.position = 0
                    self.trailing_stop = 0
                    self.start_price = 0
                
            if self.position == -1:
                if self.start_price*(1+sl_coef)<=self.data.close.iloc[bar]:
                    self.close_short(bar, size='all')
                    self.pos_result = -1
                    self.position = 0
                    self.trailing_stop = 0
                    self.start_price = 0
                elif self.start_price*(1-tp_coef)>=self.data.close.iloc[bar]:
                    self.close_short(bar, size='all')
                    self.pos_result = 1
                    self.position = 0
                    self.trailing_stop = 0
                    self.start_price = 0
                
                
         #OPEN POSITIONS
            if self.position == 0:
                if self.data.soldier_nr_neg[bar] == 2 and self.data.rng_vs_price[bar]<=0.003:
                    self.go_long(bar,amount = 'all')
                    self.position = 1
                    self.rng_vs_price_start = self.data.rng_vs_price.iloc[bar]
                    self.trailing_stop = self.data.close.iloc[bar] - (self.data.ATR.iloc[bar]*sl_coef)
                    self.start_price = self.data.close.iloc[bar]  
                    self.data['start_price'].iloc[bar] = self.start_price
                               
                elif self.data.soldier_nr_pos[bar] == 2 and self.data.rng_vs_price[bar]<=0.003:
                    self.go_short(bar,amount = 'all')
                    self.position = -1
                    self.rng_vs_price_start = self.data.rng_vs_price.iloc[bar]
                    self.trailing_stop = self.data.close.iloc[bar] + (self.data.ATR.iloc[bar]*sl_coef)
                    self.start_price = self.data.close.iloc[bar]   
                    self.data['start_price'].iloc[bar] = self.start_price
                

            self.data['position'].iloc[bar] = self.position
            self.data['trailing_stop'].iloc[bar] = self.trailing_stop
            self.data['pos_result'].iloc[bar] = self.pos_result
            self.data['rng_vs_price_start'].iloc[bar] = self.rng_vs_price_start
        self.close_all(bar+1) # close pos at the last bar       
        

sf.get_stored_data_close('BTCBUSD','1h',"2020-01-01","2022-11-28")

bc = IterativeBacktest("BTCBUSD","2022-11-01","2023-03-23",tf='1h',amount = 1000)
bc = IterativeBacktest("BTCBUSD","2023-03-01","2023-03-10",tf='1m',amount = 1000)
bc = IterativeBacktest("BTCBUSD","2023-03-01","2023-03-23",tf='5m',amount = 1000)


bc.data.rng_vs_price[(bc.data.rng_vs_price<0.005)].hist(bins=5)
plt.show()

bc.calculate_test_strategy(body_size =0.8, ma_interval = 15, sl_coef=0.01, tp_coef = 0.02)
bc.plot_data()

sf.excel_export(bc.data[['pos_result','rng_vs_price','returns','cumreturns','position','rng_vs_price_start']])

# sf.excel_export(bc.data.tail(10000))
#Find best params 
df = pd.DataFrame(columns =['combination' ,'accuracy','perf','perf_wo_comm', 'trades' ])
combination = list()
acc_list = list()
perf_list = list()
perf_w_comm_list = list()
trades_list = list()

#Iterator of best params tf and pair
ma_interval = list(range(5,25,5))
bb_interval = list(range(10,40,10))
body_size = list(np.arange(0.7,0.9,0.1))
sl_coef = list(np.arange(1,2.25,0.25))
vol_coef = list(np.arange(1,2,0.25))

all_combinations = list(itertools.product(ma_interval, bb_interval, body_size, sl_coef, vol_coef))
start_time = time.time()
counter = 1
for i in all_combinations:
    bc.calculate_onebar_strategy(ma_interval = i[0], bb_interval=i[1], body_size=i[2], sl_coef=i[3], vol_coef=i[4])
    df = df.append(
        {
        'combination':i,
        'accuracy':round(len(bc.data.loc[bc.data['pos_result']==1,['pos_result']])/len(bc.data.loc[bc.data['pos_result']!=0,['pos_result']]),2),
        'perf':(bc.current_balance - bc.initial_balance) / bc.initial_balance * 100,
        'perf_wo_comm':(bc.current_balance - bc.initial_balance - bc.costs) / bc.initial_balance * 100,
        'trades':bc.trades
        },
        ignore_index=True
    )
    print(f'{counter} / {len(all_combinations)} done...')
    counter += 1
opt = df.iloc[np.argmax(df.perf_wo_comm)]
print(75*"-")
print('The Best combination is: \n{}'.format(opt))
print(f'time spent on gridsearch in minutes: {round((time.time() - start_time)/60,2)}')

#sf.excel_export(df)
