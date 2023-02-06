import os 
#os.chdir('c:\\Users\\artjoms.jersovs\\github\\AJbots\\')
import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import stored_functions as sf
import matplotlib.pyplot as plt
import warnings
from ta.volatility import average_true_range, BollingerBands
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV
warnings.filterwarnings('ignore')

client, bsm = sf.setup_api_conn_binance()

#bc = IterativeBacktest("BTCBUSD","2020-01-01","2022-11-28",tf='1h',amount = 1000)
data = sf.get_stored_data_close('BTCBUSD', '1h', '2020-01-01','2022-11-28')


#            Feature        Gain       Cover  Frequency
# 1:            pctB 0.558285050 0.454057292 0.35208333
# 2:         low_pct 0.132363278 0.155349397 0.17083333
# 3:        high_pct 0.100599531 0.120882005 0.14791667
# 4:  high_pct_lag_1 0.068865766 0.079338732 0.11875000
# 5: close_pct_lag_1 0.046947560 0.061459141 0.05416667
# 6:   low_pct_lag_2 0.045082123 0.057099186 0.08750000
# 7:        open_pct 0.034775661 0.049199204 0.03958333
# 8:   low_pct_lag_1 0.008435471 0.015357083 0.01875000
# 9:  open_pct_lag_1 0.004645560 0.007257961 0.01041667

data['open_pct'] = data['open'].pct_change()*100
data['high_pct'] = data['high'].pct_change()*100
data['low_pct'] = data['low'].pct_change()*100
data['close_pct'] = data['close'].pct_change()*100
indicator_bb = BollingerBands(close=data["close"], window=20, window_dev=2)
data['pctB'] = indicator_bb.bollinger_pband()

lags = 5
cols = []
for lag in range(1, lags + 1):
    for subcol in data.columns[5:9]:
        subcol_name = "{}_lag_{}".format(subcol,lag)
        data[subcol_name] = data[subcol].shift(lag)
        cols.append(subcol_name)
        
data.dropna(inplace = True)
data['target'] = np.where(data.close.shift(-1)>data.close,1,0)
data = data[['target','pctB','low_pct','high_pct','high_pct_lag_1','close_pct_lag_1','low_pct_lag_2','open_pct','low_pct_lag_1','open_pct_lag_1']].copy()

loaded_model = xgb.XGBClassifier()
loaded_model.load_model('strategy_sandbox/012_one_impulse/xgb_bin_58_buy.json')
buy_model = loaded_model
#sf.excel_export(data)
X = buy_model.predict_proba(data[['pctB','low_pct','high_pct','high_pct_lag_1','close_pct_lag_1','low_pct_lag_2','open_pct','low_pct_lag_1','open_pct_lag_1']].values[-1:,])
X[:,1][0]
## SPLIT ON TRAIN AND TEST
times = sorted(data.index.values)
last_20pct = sorted(data.index.values)[-int(0.2*len(times))] # Last 10% of series
df_train = data[(data.index < last_20pct)]  # Training data are 80% of total data
df_test = data[(data.index >= last_20pct)]

X_train = df_train.drop(['target'], axis=1)
y_train = df_train['target'].astype('int32')
X_test = df_test.drop(['target'], axis=1)
y_test = df_test['target'].astype('int32')

params = {
            'objective':'binary:logistic',#'multi:softprob',
            'max_depth': 2,
            'min_child_weight':12,
            'learning_rate':0.005,
            'subsample':0.75,
            'colsample_bytree':0.75,
            'n_estimators':81
            #'eval_metric':'auc'
        }         

y_pred, proba_df, xgb_clf = sf.fit_predict_eval_bin_xgb(X_train, y_train, X_test, y_test, params = params, balance_weights=False)

proba_df['trade'] = np.where((proba_df.prob_long>=0.52),1,0)
conf = pd.crosstab(proba_df['y_test'],proba_df['trade'])
acc = conf[1][1]/(conf[1][0]+conf[1][1])
print(acc)

xgb_clf.save_model('xgb_bin_58_buy.json')


#SELL model
#rerun data processing
data['target'] = np.where(data.close.shift(-1)<data.close,1,0)
data.dropna(inplace = True)
data = data[['target','pctB','low_pct','high_pct','high_pct_lag_1','close_pct_lag_1','low_pct_lag_2','open_pct','low_pct_lag_1','open_pct_lag_1']].copy()

params = {
            'objective':'binary:logistic',#'multi:softprob',
            'max_depth': 2,
            'min_child_weight':15,
            'learning_rate':0.005,
            'subsample':0.9,
            'colsample_bytree':0.75,
            'n_estimators':160
            #'eval_metric':'auc'
        }         

y_pred, proba_df, xgb_clf = sf.fit_predict_eval_bin_xgb(X_train, y_train, X_test, y_test, params = params, balance_weights=False)

proba_df['trade'] = np.where((proba_df.prob_long>0.52),1,0)
conf = pd.crosstab(proba_df['y_test'],proba_df['trade'])
acc = conf[1][1]/(conf[1][0]+conf[1][1])
print(acc)

xgb_clf.save_model('xgb_bin_58_sell.json')