#ADD ALL NECESSARY LIBRARIES
import os
import numpy as np
import pandas as pd
import binance as bin
import itertools
import requests 
import json 
import datetime as dt
import matplotlib.pyplot as plt
import stored_functions as sf
from ta.volatility import average_true_range
plt.style.use("seaborn")

class Backtester():
    def __init__(self, symbol, atr_period, multiplicator, TP_atr, SL_atr, start, end, tf = '15m'):

        #DEFINE MAIN PARAMETERS
        self.symbol = symbol

        #******DEFINE ALL VARIABLES FOR CHALLENGE*************
        self.atr_period = atr_period
        self.multiplicator = multiplicator
        self.TP_atr = TP_atr
        self.SL_atr = SL_atr
        #****************************************************
        self.start = start
        self.end = end 
        self.tf = tf 
        self.results = None 
        self.df = self.preprocess_data()

    def __repr__(self):
        rep = "Backtester(symbol = {}, atr_period = {}, multiplicator = {}, TP_atr = {}, SL_atr = {}, start = {}, end = {}"
        return rep.format(self.symbol, self.atr_period, self.multiplicator, self.TP_atr, self.SL_atr, self.start, self.end)

    def preprocess_data(self):
        filename = str(os.getcwd())+'\\datasets\\%s-1m-data.csv' % (self.symbol)
        
        if os.path.isfile(filename):
            raw = pd.read_csv(filename)
            raw['timestamp'] = pd.to_datetime(raw['timestamp'])
            raw.set_index('timestamp', inplace=True)
            raw = raw.resample(self.tf, label = 'right').last().ffill().apply(pd.to_numeric)
            print('All caught up! Existing dataset is used.')
        else:
            raw = sf.get_all_binance(self.symbol, self.tf, save=True)
        
        df = raw.loc[str(self.start):str(self.end)].copy()
        df = df.dropna().apply(pd.to_numeric)

        #******ADD NECESSARY INDICATORS*************
        df['returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
        df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.high, window = self.atr_period)

        #****************************************************
        df.dropna(inplace=True)
        self.data = df 
        return print(self.data.head(10))

        #******DEFINE ALL VARIABLES AS INPUT PARAMS*************
    def set_params(self, atr_period = None, multiplicator = None,  TP_atr = None, SL_atr = None):
        if atr_period is not None:
            self.atr_period = atr_period
            self.data['ATR'] = average_true_range(high = self.data.high, low = self.data.low, close = self.data.high, window = self.atr_period)
        
        if multiplicator is not None:
            self.multiplicator = multiplicator
        
        if TP_atr is not None:
            self.TP_atr = TP_atr

        if SL_atr is not None:
            self.SL_atr = SL_atr
        #****************************************************

    def test_strategy(self):
        data = self.data
        #data['ATR'] = average_true_range(high = data.high, low = data.low, close = data.close, window = self.atr_period)
        data['position'] = np.where((data.close-data.open> data.ATR*self.multiplicator) & data.ATR>0 , 1, np.nan)
        data['position_open_price']= data.position*data.close
        data['position_open_price'] = data.position_open_price.fillna(method='ffill')
        data['position'] = np.where((data.close - data.position_open_price>data.position_open_price*(self.TP_atr+1)) , 0, data.position)
        data['position'] = np.where((data.close < data.position_open_price * (1-self.SL_atr) ), 0, data.position)
        data['position'] = data.position.ffill(axis=0).fillna(0)
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

            bench_monthly_return = self.results.returns.mean() * 2880 #returns in % monthly
            bench_monthly_var = self.results.returns.std() * np.sqrt(2880) #risk in % monthly #make sense only in normally distributed returns as std and mean is used
            bench_max_drawdown = self.results.bench_drawdown.max()

            strategy_monthly_return = self.results.strategy.mean() * 2880
            strategy_monthly_var = self.results.strategy.std() * np.sqrt(2880) 
            strategy_max_drawdown = self.results.strategy_drawdown.max()
            print(75 * "-")
            print('{}% - Benchmark monthly return'.format(round(bench_monthly_return*100,2)))
            print('{}% - Strategy monthly return'.format(round(strategy_monthly_return*100,2)))
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
    def optimize_params(self, ATR_range, Multip_range, TP_range, SL_range):
        results = []
        combinations = list(itertools.product(range(*ATR_range),np.arange(*Multip_range),np.arange(*TP_range),np.arange(*SL_range)))
        print('{} combinations will be tested. Wait please!'.format(len(combinations)))
        for i in combinations:
            self.set_params(i[0],i[1],i[2],i[3])
            results.append(self.test_strategy()[0])
        best_perf = np.max(results)
        opt = combinations[np.argmax(results)]

        self.set_params(opt[0],opt[1], opt[2], opt[3])
        self.test_strategy()

        many_results = pd.DataFrame(data = combinations, columns = ['ATR','Multiplicator', 'TP_atr', 'SL_atr'])
        many_results['performance'] = results
        self.optimization_results = many_results
        return opt, best_perf


test = Backtester('BTCUSDC',10,1.1,0.01,0.01,'2021-01-01','2022-02-01', '15m')

test.test_strategy()
test.plot_results()
test.optimize_params((10,30,5),(1.2,2,0.1),(0.01,0.05,0.002),(0.01,0.05,0.002))
test.plot_strategy()
test.set_params(atr_period=10,multiplicator=1.3,TP_atr=0.04, SL_atr=0.01) #best params for 2021 for BTCUSDT

test.optimization_results
sf.excel_export(test.results)

#test.data.to_csv("BTCUSDT_15m_JAN21_NOV21.csv")

test.results

test.results.tail(1000)


sf.excel_export(test.results)
sf.excel_export(test.optimization_results)