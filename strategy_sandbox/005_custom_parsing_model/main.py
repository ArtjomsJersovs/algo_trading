from audioop import avg
from math import nan
import os
import time
import datetime as dt
from binance.helpers import date_to_milliseconds
import numpy as np
import itertools
from numpy import NaN, dtype, std
from numpy.lib import histograms
import pandas as pd
import json 
from binance import ThreadedWebsocketManager
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range
from ta import add_all_ta_features
import stored_functions as sf
import subprocess
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV, train_test_split
from matplotlib import pyplot
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# sf.clearvars()


pd.set_option('display.max_rows', 100)
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

#GET DATA

h1_data = sf.get_all_binance("BTCUSDT","1h", save=True)
h1_data['timestamp'] = pd.to_datetime(h1_data.index)
h1_data.set_index('timestamp', inplace=True)
h4_data = h1_data.resample('4h', label = 'right').last().ffill().apply(pd.to_numeric)
#h4_data = h4_data['2022']

d1_data = sf.get_all_binance("BTCUSDT","1d", save=True)
d1_data['timestamp'] = pd.to_datetime(d1_data.index)
d1_data.set_index('timestamp', inplace=True)
#d1_data = d1_data['2022']


df_list = [h4_data, d1_data]
sma_list = list(range(5,105,5))
sma_list_names = ['SMA_'+str(sub) for sub in sma_list]

for df in df_list:
    df['returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
    df['returns_lag'] = df.returns.shift(-1) 
    for sma in sma_list:
        df[f'SMA_{sma}'] = df.close.rolling(sma).mean()
    
    for each_sma, compare_sma in list(itertools.combinations(list(df.filter(like='SMA_'))[:-1],2)):
        df[f'{each_sma}-{compare_sma}'] = df[f'{each_sma}'] - df[f'{compare_sma}'] 
  
    #generate additional features for both datasets
    df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.close, window = 14)
    df['target_l_tmp'] = np.where((df.high.shift(1)<df.close) & (df.returns>0) & ((df.close-df.open)>df.ATR),1,0)
    df['target_s_tmp'] = np.where((df.low.shift(1)>df.close) & (df.returns<0) & ((df.open-df.close)>df.ATR),-1,0)
    df['target'] = np.where(df.target_l_tmp.shift(-1)==1, 1,
                   np.where(df.target_s_tmp.shift(-1)==-1, -1, 0))

print('Done!')   



h4_data_clean = h4_data.drop(['open','high','low','quote_av','trades','tb_base_av','tb_quote_av','close_time','ignore','target_l_tmp','target_s_tmp','returns'], axis=1)
h4_data_clean = h4_data_clean.iloc[100:,].iloc[:-1]
d1_data_clean = d1_data.drop(['open','high','low','quote_av','trades','tb_base_av','tb_quote_av','close_time','ignore','target_l_tmp','target_s_tmp','returns'], axis=1)
d1_data_clean = d1_data_clean.iloc[100:,].iloc[:-1]
vars_to_keep = ["SMA_50", "SMA_5-SMA_15", "SMA_5-SMA_50", "SMA_25-SMA_35", "SMA_40-SMA_45", "SMA_55-SMA_85", "SMA_90-SMA_95", "ATR", "volume", "target","returns_lag", "close"]
h4_data_clean = h4_data_clean[vars_to_keep]
d1_data_clean = d1_data_clean[vars_to_keep]

# sns.heatmap(h4_data_clean.corr())
# plt.show()
# d1_data.groupby(['target']).describe()
# sf.excel_export(h4_data_clean)


df_main = d1_data_clean
validation_returns = df_main.returns_lag
df_main = df_main.apply(pd.to_numeric).copy().drop(['returns_lag'], axis=1)


df_main.target = df_main.target.replace(1,2).replace(-1,1).replace(np.nan,0)
#replace all na's to 999
df_main = df_main.fillna(-999).replace(np.inf,-999).replace(-np.inf,-999)
#split on target and predictors
X = df_main.drop('target', axis=1)
y = df_main.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# declare parameters
params = {
            'objective':'multi:softprob',
            'max_depth': 10,
            'min_child_weight':15,
            'num_class': 3,
            'alpha': 0.1,
            'learning_rate':0.5,
            'n_estimators':500
        }         
    
y_pred, proba_df, xgb_clf = sf.fit_predict_eval_multi_xgb(X_train, y_train, X_test, y_test, params)          


#K-fold validation
# params2 = {
#             'objective':'multi:softprob',
#             'max_depth': 20,
#             'min_child_weight':10,
#             'num_class': 3,
#             'alpha': 0.1,
#             'learning_rate':0.5
#             #'n_estimators':60
#         }             
# data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
# xgb_clf = xgb.cv(dtrain=data_dmatrix, params=params2, nfold=5, metrics = ['mlogloss','merror'],seed=42, num_boost_round = 50) 
# print(xgb_clf.iloc[-1])


ret_lag = X_test.join(validation_returns)
dfc = pd.DataFrame({'y_test':y_test,'y_pred':y_pred,'prob_short':proba_df.prob_short, 'prob_long':proba_df.prob_long, 'future_return':ret_lag.returns_lag})

#ACCURACY BY RETURNS
# dfc.y_pred = np.where(((dfc.prob_short-dfc.prob_long)>0.5) & (np.sign(dfc.future_return)==-1),-1,
#              np.where(((dfc.prob_long-dfc.prob_short)>0.5) & (np.sign(dfc.future_return)==-1),1,0))
dfc['match_flag'] = np.where(((dfc.y_pred==1) & (np.sign(dfc.future_return)==-1))|((dfc.y_pred==2) & (np.sign(dfc.future_return)==1)),1,0)

print('Accuracy by future returns is: {}%'.format((dfc.match_flag.sum()/dfc[dfc.y_pred!=0].shape[0])*100))

#sf.excel_export(dfc)

