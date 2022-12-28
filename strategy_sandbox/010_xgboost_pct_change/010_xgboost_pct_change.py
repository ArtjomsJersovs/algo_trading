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

data = sf.get_stored_data_close('BTCBUSD', '1h', '2022-06-01','2022-11-28')


#PREPROCESS FEATURES

data['open'] = data['open'].pct_change()
data['high'] = data['high'].pct_change()
data['low'] = data['low'].pct_change()
data['close'] = data['close'].pct_change()
data['volume'] = data['volume'].pct_change()
data['target'] =np.where(data.close>0,1,0)
data = data.dropna()
# ##NORMALIZATION
# min_return = min(data[['open', 'high', 'low', 'close']].min(axis=0))
# max_return = max(data[['open', 'high', 'low', 'close']].max(axis=0))

# # Min-max normalize price columns (0-1 range)
# data['open'] = (data['open'] - min_return) / (max_return - min_return)
# data['high'] = (data['high'] - min_return) / (max_return - min_return)
# data['low'] = (data['low'] - min_return) / (max_return - min_return)
# data['close'] = (data['close'] - min_return) / (max_return - min_return)

# min_volume = data['volume'].min(axis=0)
# max_volume = data['volume'].max(axis=0)

# # Min-max normalize volume columns (0-1 range)
# data['volume'] = (data['volume'] - min_volume) / (max_volume - min_volume)

lags = 5
cols = []
cols.append('target')
for lag in range(1, lags + 1):
    for subcol in data.columns[1:5]:
        subcol_name = "{}_lag{}".format(subcol,lag)
        data[subcol_name] = data[subcol].shift(lag)
        cols.append(subcol_name)
data.dropna(inplace = True)
data = data[cols]

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
            'max_depth': 5,
            'min_child_weight':10,
            'alpha': 1,
            'learning_rate':0.01,
            'n_estimators':100
            #'eval_metric':'auc'
        }         

y_pred, proba_df, xgb_clf = sf.fit_predict_eval_bin_xgb(X_train, y_train, X_test, y_test, params = params, balance_weights=False)

proba_df['trade'] = np.where((proba_df.prob_short>0.65),-1,0)
proba_df['trade'] = np.where((proba_df.trade==0)&(proba_df.prob_long>0.65),1,proba_df.trade)

proba_df['trade'] = np.where(abs(proba_df.prob_short-proba_df.prob_long)>0.15,y_pred,np.nan)
proba_df['trade'] = np.where((proba_df.trade==0)&(proba_df.prob_short<=0.3)&(proba_df.prob_long>0.7),1,proba_df.trade)

accuracy = np.where((proba_df.trade==1) & (proba_df.y_test==1),1,0)
accuracy = np.where((proba_df.trade==-1)&(proba_df.y_test==0),1,accuracy)

sf.excel_export(proba_df)


#dimensions