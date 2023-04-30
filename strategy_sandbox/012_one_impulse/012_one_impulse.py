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
import seaborn as sns
import itertools
plt.style.use("ggplot")


pd.set_option('display.max_rows', 100)
#INIT API
client, bsm = sf.setup_api_conn_binance()

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

              
    def calculate_onebar_strategy(self, ma_interval=10, bb_interval=20, body_size =0.8, sl_coef=1, vol_coef = 1):
        # reset all parameters
        self.position = 0
        self.trades = 0
        self.costs = 0
        self.current_balance = self.initial_balance
        self.trailing_stop = 0
        self.start_price = 0
        self.pos_result = 0
        
        self.data['ATR'] = average_true_range(high = self.data.high, low = self.data.low, close = self.data.close, window = ma_interval)
        indicator_bb = BollingerBands(close=self.data["close"], window=bb_interval, window_dev=2)
        self.data['close_ma'] = self.data['close'].rolling(ma_interval).mean()
        self.data['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.data['bb_bbh'] = indicator_bb.bollinger_hband()
        self.data['bb_bbl'] = indicator_bb.bollinger_lband()
        self.data['volume_ma'] = self.data['volume'].rolling(ma_interval).mean()
        self.data['position'] = 0
        self.data['trailing_stop'] = 0
        self.data['pos_result'] = 0
        self.data['start_price'] = 0
        #strategy specific
        self.data['higher_low'] = np.where(self.data.low>self.data.low.shift(1),1,0)
        self.data['lower_high'] = np.where(self.data.high<self.data.high.shift(1),1,0)
        self.data['higher_close'] = np.where(self.data.close>self.data.close.shift(1),1,0)
        self.data['lower_close'] = np.where(self.data.close<self.data.close.shift(1),1,0)
        self.data['higher_volume'] = np.where(self.data.volume>self.data.volume.shift(1),1,0)
        self.data['over_ma_volume'] = np.where(self.data.volume>(self.data.volume_ma * vol_coef),1,0)
        
        self.data['rng'] = abs(self.data.high-self.data.low)
        self.data['rng_body'] = abs(self.data.open-self.data.close)
        self.data['body_size'] = np.where(self.data.rng_body.div(self.data.rng)>body_size,1,0)

        self.data['returns'] = np.log(self.data.close.astype(float).div(self.data.close.astype(float).shift(1)))
        self.data['cumreturns'] = self.data['returns'].cumsum().apply(np.exp)
        #self.data.dropna(inplace = True)
        
        for bar in tqdm(range(len(self.data)-1)): # all bars except of last one
        #for bar in range(len(self.data)-1):
            self.pos_result = 0
        #OPEN POSITIONS
            if self.position == 0:
                if self.data.higher_low.iloc[bar]==1 and self.data.higher_close.iloc[bar]==1 and self.data.body_size.iloc[bar]==1 and self.data.over_ma_volume.iloc[bar]==1 and (self.data.bb_bbl.iloc[bar]<self.data.close.iloc[bar]) and (self.data.bb_bbh.iloc[bar]>self.data.close.iloc[bar]):
                    self.go_long(bar,amount = 'all')
                    self.position = 1
                    self.trailing_stop = self.data.close.iloc[bar] - (self.data.ATR.iloc[bar]*sl_coef)
                    self.start_price = self.data.close.iloc[bar]  
                    self.data['start_price'].iloc[bar] = self.start_price
                               
                elif self.data.lower_high.iloc[bar]==1 and self.data.lower_close.iloc[bar]==1 and self.data.body_size.iloc[bar]==1 and self.data.over_ma_volume.iloc[bar]==1 and (self.data.bb_bbh.iloc[bar]>self.data.close.iloc[bar]) and (self.data.bb_bbl.iloc[bar]<self.data.close.iloc[bar]):
                    self.go_short(bar,amount = 'all')
                    self.position = -1
                    self.trailing_stop = self.data.close.iloc[bar] + (self.data.ATR.iloc[bar]*sl_coef)
                    self.start_price = self.data.close.iloc[bar]   
                    self.data['start_price'].iloc[bar] = self.start_price
                    
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
                    self.trailing_stop = self.data.close.iloc[bar] - (self.data.ATR.iloc[bar]*sl_coef)
                    
                    
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
                    self.trailing_stop = self.data.close.iloc[bar] + (self.data.ATR.iloc[bar]*sl_coef)

            self.data['position'].iloc[bar] = self.position
            self.data['trailing_stop'].iloc[bar] = self.trailing_stop
            self.data['pos_result'].iloc[bar] = self.pos_result
        self.close_all(bar+1) # close pos at the last bar       
        


sf.get_stored_data_close('ATOMUSDT','1h',"2023-01-01","2023-04-25")


bc = IterativeBacktest("DOTBUSD","2022-11-01","2023-04-15",tf='1h',amount = 1000)

sf.excel_export(bc.data)


bc.calculate_onebar_strategy(ma_interval=20, bb_interval=20, body_size =0.7, sl_coef=1.5, vol_coef = 1) # NEARBUSD (20, 20, 0.7, 1.5, 1.0)
bc.calculate_onebar_strategy(ma_interval=20, bb_interval=30, body_size =0.8, sl_coef=1, vol_coef = 1.25) # ATOMBUSD (20, 30, 0.7999999999999999, 1.0, 1.25)
bc.calculate_onebar_strategy(ma_interval=15, bb_interval=20, body_size =0.7, sl_coef=1.5, vol_coef = 1.25) # ADABUSD (15, 20, 0.7, 1.5, 1.25) 

#worser performers - keep only three
bc.calculate_onebar_strategy(ma_interval=20, bb_interval=40, body_size =0.7, sl_coef=2, vol_coef = 1.5) # DOTBUSD (20, 40, 0.7, 2.0, 1.5)
bc.calculate_onebar_strategy(ma_interval=10, bb_interval=20, body_size =0.7, sl_coef=2, vol_coef = 1.5) #BTCBUSD (10, 20, 0.7, 2.0, 1.5)
bc.calculate_onebar_strategy(ma_interval=15, bb_interval=20, body_size =0.7, sl_coef=2, vol_coef = 1.25) # ETHBUSD (15, 20, 0.7, 2.0, 1.25)

bc.plot_data()
bc.data.tail(5)



#Plot trades overlap across all pairs
df_pos_tickers = pd.DataFrame(bc.data.loc[:,('position')].rename(str('btc')))
add_series = bc.data.loc[:,('position')].rename(str('dot'))
df_pos_tickers = df_pos_tickers.assign(dot=add_series)
df_pos_tickers.replace(-1,1, inplace=True)
#heatmap = sns.heatmap(df_pos_tickers.corr(), vmin=-1, vmax=1, annot=True)


#Find best params 
df = pd.DataFrame(columns =['combination' ,'accuracy','perf','perf_wo_comm', 'trades' ])
combination = list()
acc_list = list()
perf_list = list()
perf_w_comm_list = list()
trades_list = list()

#Iterator of best params tf and pair
ma_interval = list(range(5,25,5))
bb_interval = list(range(20,50,10))
body_size = list(np.arange(0.7,0.9,0.1))
sl_coef = list(np.arange(1,2.25,0.5))
vol_coef = list(np.arange(1,1.75,0.25))

all_combinations = list(itertools.product(ma_interval, bb_interval, body_size, sl_coef, vol_coef))
pairs = ['ATOMUSDT','NEARBUSD','DOTBUSD','ADABUSD']
# for tick in pairs:
#     sf.get_stored_data_close(tick,'1h',"2022-11-01","2023-04-15")


for ticker in pairs:
    bc = IterativeBacktest(ticker,"2022-11-01","2023-04-15",tf='1h',amount = 1000)
    df = pd.DataFrame(columns =['combination' ,'accuracy','perf','perf_wo_comm', 'trades' ])
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
    df.to_csv('C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/production/012_one_bar/{}_test_results.csv'.format(ticker))

#cron incremental logs
#/c "powershell.exe py C:\Users\Administrator\Documents\algo_trading\production\012_cron_one_bar\104_012_one_bar_strategy_cron_dot.py" >>C:\Users\Administrator\Documents\algo_trading\production\012_cron_one_bar\104_012_dot_1h_log.log