#ADD ALL NECESSARY LIBRARIES
import os
import numpy as np
import pandas as pd
import binance as bin
import itertools
import subprocess
import requests 
import json 
import datetime as dt
import stored_functions as sf
import matplotlib.pyplot as plt
from ta.volatility import average_true_range
from ta.trend import MACD
plt.style.use("seaborn")

def excel_export(df, name='temp_file', size=100000):
    df.head(int(size)).to_excel(str(name) +".xlsx") 
    subprocess.run(["C:/Program Files/Microsoft Office/root/Office16/EXCEL.exe", str(name) +".xlsx"])



class Backtester():
    def __init__(self, symbol, fast_ma, slow_ma, smooth, start, end, tf = '15m'):

        #DEFINE MAIN PARAMETERS
        self.symbol = symbol

        #******DEFINE ALL VARIABLES FOR CHALLENGE*************
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.smooth = smooth
        #****************************************************
        self.start = start
        self.end = end 
        self.tf = pd.to_timedelta(str(tf).replace('m','min'))
        self.results = None 
        self.df = self.preprocess_data()

    def __repr__(self):
        rep = "Backtester(symbol = {}, slow_ma = {}, fast_ma = {}, smooth = {}, start = {}, end = {}"
        return rep.format(self.symbol, self.slow_ma, self.fast_ma, self.smooth, self.start, self.end)

    def preprocess_data(self):
        filename = str(os.getcwd())+'\\datasets\\%s-1m-data.csv' % (self.symbol)

        
        if os.path.isfile(filename):
            raw = pd.read_csv(filename)
            raw['timestamp'] = pd.to_datetime(raw['timestamp'])
            raw.set_index('timestamp', inplace=True)
            raw = raw.resample(self.tf, label = 'right').last().ffill().apply(pd.to_numeric)
            print('All caught up! Existing dataset is used.')
        else:
            raw = sf.get_all_binance(self.symbol, self.tf, save=False)
        
        df = raw.loc[str(self.start):str(self.end)].copy()
        df = df.dropna().apply(pd.to_numeric)

        #******ADD NECESSARY INDICATORS*************
        df['returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))

        macd_obj = MACD(df.close, self.slow_ma,self.fast_ma,self.smooth)
        df['MACD'] = macd_obj.macd()
        df['MACD_hist'] = macd_obj.macd_diff()
        df['MACD_signal'] = macd_obj.macd_signal()

        #****************************************************
        df.dropna(inplace=True)
        self.data = df 
        return None

        #******DEFINE ALL VARIABLES AS INPUT PARAMS*************
    def set_params(self, fast_ma = None, slow_ma = None,  smooth = None):
        self.slow_ma = slow_ma
        self.fast_ma = fast_ma
        self.smooth = smooth
        macd_obj = MACD(self.data.close, self.slow_ma,self.fast_ma,self.smooth)
        self.data['MACD'] = macd_obj.macd()
        self.data['MACD_hist'] = macd_obj.macd_diff()
        self.data['MACD_signal'] = macd_obj.macd_signal()
        
        #****************************************************

    def test_strategy(self):
        data = self.data
        data['position'] = np.where((np.sign(data.MACD)<0) & (data.MACD>data.MACD_signal) & (data.MACD.shift(1)<data.MACD_signal.shift(1)),1,np.nan)
        data['position'] = np.where((np.sign(data.MACD)>0) & (data.MACD<data.MACD_signal) & (data.MACD.shift(1)>data.MACD_signal.shift(1)),0,data.position)
        data['position']  = data.position.ffill().fillna(0)
        data['trades'] = np.where((data.position!=data.position.shift(1)),1,0)
        data['trades'] = data.trades.cumsum()-1
        data['strategy'] = data['position'].shift(1) * data['returns']
        data['cumstrategy'] = data['strategy'].cumsum().apply(np.exp)
        data['cumreturns'] = data['returns'].cumsum().apply(np.exp)

    ## Performance metrics calculations ##
    ##**********************************##

        data['bench_cummax'] = data.cumreturns.cummax() 
        data['bench_drawdown'] = (data['bench_cummax'] - data['cumreturns']) / data['bench_cummax']
        
        data['strategy_cummax'] = data.cumstrategy.cummax() 
        data['strategy_drawdown'] = (data['strategy_cummax'] - data['cumstrategy']) / data['strategy_cummax']

        perf = round(data['cumstrategy'].iloc[-1]*100,2)
        benchmark = round(data['cumreturns'].iloc[-1]*100,2)
        self.results = data 
        return perf, benchmark

    def plot_results(self):
        if self.results is None:
            print('Run test_strategy() first!')
        else:
            title = '{} - Parameter tuning results'.format(self.symbol)
            self.results[['cumreturns','cumstrategy']].plot(title = title, figsize=(12, 8))

            month = pd.to_timedelta(str('30d'))

            bench_monthly_return = self.results.returns.mean() * (month/self.tf) #returns in % monthly
            bench_monthly_var = self.results.returns.std() * np.sqrt(month/self.tf) #risk in % monthly #make sense only in normally distributed returns as std and mean is used
            bench_max_drawdown = self.results.bench_drawdown.max()

            strategy_monthly_return = self.results.strategy.mean() * (month/self.tf)
            strategy_monthly_var = self.results.strategy.std() * np.sqrt(month/self.tf) 
            strategy_max_drawdown = self.results.strategy_drawdown.max()
            print(75 * "-")
            print('{}% - Benchmark monthly return'.format(round(bench_monthly_return*100,2)))
            print('{}% - Strategy monthly return'.format(round(strategy_monthly_return*100,2)))
            print('{} trades initiated, in total: {}% on commissions'.format(self.data['trades'].iloc[-1],(self.data['trades'].iloc[-1])*0.075))
            print(75 * "-")
            print('{}% - Benchmark monthly VaR'.format(round(bench_monthly_var*100,2)))
            print('{}% - Strategy monthly VaR'.format(round(strategy_monthly_var*100,2)))
            print(75 * "-")
            print('{}% - Benchmark Maximum drawdown on {}'.format(round(bench_max_drawdown*100,2),self.results.bench_drawdown.idxmax()))
            print('{}% - Strategy Maximum drawdown on {}'.format(round(strategy_max_drawdown*100,2),self.results.strategy_drawdown.idxmax()))
            print(75 * "-")
        return plt.show()

    def plot_strategy(self):
        if self.results is None:
            print('Run test_strategy() first!')
        else:
            title = '{} - Parameter tuning results'.format(self.symbol)
            self.results[['close','position']].plot(title = title, figsize=(12, 8), secondary_y = 'position')
        return plt.show()

    #******DEFINE ALL VARIABLES AS INPUT PARAMS*************
    def optimize_params(self,fast_ma, slow_ma, smooth):
        results = []
        bench = []
        combinations = list(itertools.product(range(*fast_ma),np.arange(*slow_ma),np.arange(*smooth)))
        print('{} combinations will be tested. Wait please!'.format(len(combinations)))
        for i in combinations:
            self.set_params(i[0],i[1],i[2])
            results.append(self.test_strategy()[0])
            bench.append(self.test_strategy()[1])
        best_perf = np.max(results)
        opt = combinations[np.argmax(results)]

        self.set_params(opt[0],opt[1], opt[2])
        self.test_strategy()

        many_results = pd.DataFrame(data = combinations, columns = ['Fast MA','Slow MA', 'Smooth'])
        many_results['performance'] = results
        many_results['bench'] = bench
        self.optimization_results = many_results
        return opt, best_perf




#Iterator of best params tf and pair
pairs = ['LINKUSDT','BTCUSDT','ETHUSDT','ATOMUSDT','LTCUSDT']   #,'ETHUSDC','ATOMUSDC','LINKUSDC','LTCUSDC'
tfs = ['5m','15m','30m','1h','2h','3h']
pairs_tfs_combinations = list(itertools.product(pairs,tfs))
best_combinations = pd.DataFrame()
for i in pairs_tfs_combinations:
    backtest = Backtester(i[0],12,26,9,'2021-01-01','2022-04-01', i[1])
    backtest.optimize_params((5,50,5),(5,50,5),(1,9,2))
    backtest.optimization_results['Symbol'] = i[0]
    backtest.optimization_results['Timeframe'] = i[1]
    best_combinations = best_combinations.append(backtest.optimization_results, ignore_index=False)
    best_combinations.reset_index()
opt = best_combinations.iloc[np.argmax(best_combinations.performance)]
print(75*"-")
print('The Best combination is: \n{}'.format(opt))



test = Backtester('ATOMUSDT',10,15,7,'2021-01-01','2022-04-01', '1h')
test = Backtester('LINKUSDT',15,40,7,'2021-01-01','2022-04-01', '30m')
test = Backtester('ETHUSDT',10,5,3,'2020-04-01','2022-04-01', '5m')
test.test_strategy()
#test2.plot_strategy()
test.plot_results()
#sf.excel_export(test2.data)
#sf.excel_export(best_combinations)



