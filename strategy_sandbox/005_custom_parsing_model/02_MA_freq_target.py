sf.clearvars()

from audioop import avg
from math import nan
import os
import time
import datetime as dt
import json 
from binance.helpers import date_to_milliseconds
import numpy as np
import itertools
from numpy import NaN, dtype, std
from numpy.lib import histograms
import pandas as pd
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

# 

### PLAN IDEI:
### Ispoljzovatj toljko 4h i dnevnije timeframes
### Zagruzitj osnovnije parametry:
# - vyshe ili nizhe chem SMA
# - kakaja raznica mezhdu kazhdoj is SMA (v plane rastojanija)
# probuju normalizaciju - nerabotaet
# probuju otchistitj lishnije kolonny s pomoshju corr clusters i IV - toljko huzhe perf
# vsja problema v fichah - SMA bespolezny / takzhe nuzhno dobavitj cross kursi, drugije indicatori / uvelichitj kolichestvo targetov / poprobovatj D1 a ne H4

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
h1_data = h1_data.apply(pd.to_numeric)
h2_data = h1_data.resample('2h', label = 'right').last().ffill().apply(pd.to_numeric)
h4_data = h1_data.resample('2h', label = 'right').last().ffill().apply(pd.to_numeric)
#h4_data = h4_data['2022']


m15_data = sf.get_all_binance("BTCUSDT","15m", save=True)
m15_data['timestamp'] = pd.to_datetime(m15_data.index)
m15_data.set_index('timestamp', inplace=True)


df_list = [m15_data,h1_data, h2_data, h4_data]
sma_list = list(range(5,105,5))
sma_list_names = ['SMA_'+str(sub) for sub in sma_list]

for df in df_list:
    df['returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
    df['returns_lag'] = df.returns.shift(-1) 
    # for sma in sma_list:
    #     df[f'SMA_{sma}'] = df.close.rolling(sma).mean()
    
    # for each_sma, compare_sma in list(itertools.combinations(list(df.filter(like='SMA_'))[:-1],2)):
    #     df[f'{each_sma}-{compare_sma}'] = df[f'{each_sma}'] - df[f'{compare_sma}'] 
  
    #generate additional features for both datasets
    df['ATR'] = average_true_range(high = df.high, low = df.low, close = df.close, window = 14)
    df['Vol_MA'] = df.volume.rolling(15).mean()
    df['Vol_MA_1_5x'] = np.where(df.volume > (df.Vol_MA*1.5),1,0)
    df['Vol_MA_2x'] = np.where(df.volume > (df.Vol_MA*2),1,0)
    df['Vol_MA_3x'] = np.where(df.volume > (df.Vol_MA*3),1,0)
    df['MA_12'] = df.volume.rolling(12).mean()
    df['MA_26'] = df.volume.rolling(26).mean()
    df['MA_cross'] = np.where((df.MA_12.shift(1)<df.MA_26.shift(1))&(df.MA_12>df.MA_26),1,0)
    df['target_l_tmp'] = np.where((df.returns>0)& ((df.close-df.open)>(df.ATR*0.5)),1,0)
    df['target'] = df.target_l_tmp.shift(-1)
           

print('Done!')   

#sf.excel_export(h4_data)
m15_data_clean = m15_data.drop(['quote_av','trades','tb_base_av','tb_quote_av','close_time','ignore','target_l_tmp','returns'], axis=1)
m15_data_clean = m15_data_clean.iloc[26:,].iloc[:-1]
h1_data_clean = h1_data.drop(['quote_av','trades','tb_base_av','tb_quote_av','close_time','ignore','target_l_tmp','returns'], axis=1)
h1_data_clean = h1_data_clean.iloc[26:,].iloc[:-1]
h2_data_clean = h2_data.drop(['quote_av','trades','tb_base_av','tb_quote_av','close_time','ignore','target_l_tmp','returns'], axis=1)
h2_data_clean = h2_data_clean.iloc[26:,].iloc[:-1]
h4_data_clean = h4_data.drop(['quote_av','trades','tb_base_av','tb_quote_av','close_time','ignore','target_l_tmp','returns'], axis=1)
h4_data_clean = h4_data_clean.iloc[26:,].iloc[:-1]
# vars_to_keep = ["SMA_50", "SMA_5-SMA_15", "SMA_5-SMA_50", "SMA_25-SMA_35", "SMA_40-SMA_45", "SMA_55-SMA_85", "SMA_90-SMA_95", "ATR", "volume", "target","returns_lag", "close"]
# m15_data_clean = m15_data_clean[vars_to_keep]
# h1_data_clean = h1_data_clean[vars_to_keep]
# h2_data_clean = h2_data_clean[vars_to_keep]
# h4_data_clean = h4_data_clean[vars_to_keep]

df_main = m15_data_clean['2022-08':]
df_main = df_main.apply(pd.to_numeric).copy()
df_main.target.value_counts()

#df_main.target = df_main.target.replace(1,2).replace(-1,1).replace(np.nan,0)
#replace all na's to 999
df_main = df_main.fillna(-999).replace(np.inf,-999).replace(-np.inf,-999)
#split on target and predictors
X = df_main.drop('target', axis=1)
y = df_main.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

validation_returns = X_test.returns_lag
X_train=X_train.drop('returns_lag', axis=1)
X_test=X_test.drop('returns_lag', axis=1)

# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 5,
            'min_child_weight':15,
            'alpha': 0.2,
            'learning_rate':0.5,
            'n_estimators':30,
            'colsample_bytree':0.5
        }        
    
y_pred, proba_df, xgb_clf = sf.fit_predict_eval_bin_xgb(X_train, y_train, X_test, y_test, params, balance_weights=True)          


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



dfc = pd.DataFrame({'y_test':y_test,'y_pred':y_pred,'prob_short':proba_df.prob_short, 'prob_long':proba_df.prob_long, 'future_return':validation_returns})

#ACCURACY BY RETURNS FOR MULTICLASS
# dfc.y_pred = np.where(((dfc.prob_short-dfc.prob_long)>0.5),1,
#              np.where(((dfc.prob_long-dfc.prob_short)>0.5),2,0))
# dfc['match_flag'] = np.where(((dfc.y_pred==1) & (np.sign(dfc.future_return)==-1))|((dfc.y_pred==1) & (np.sign(dfc.future_return)==1)),1,0)
#print('Accuracy by future returns is: {}%'.format((dfc.match_flag.sum()/dfc[dfc.y_pred>0].shape[0])*100))

#ACCURACY BY RETURNS FOR BINARY CLASS
dfc.y_pred = np.where((dfc.prob_long>0.8)&(dfc.prob_long>dfc.prob_short),1,0)
dfc['match_flag'] = np.where((dfc.y_pred==1) & (np.sign(dfc.future_return)==1),1,0)
print('Accuracy by future returns is: {}%'.format((dfc.match_flag.sum()/dfc[dfc.y_pred==1].shape[0])*100))

#sf.excel_export(dfc)
