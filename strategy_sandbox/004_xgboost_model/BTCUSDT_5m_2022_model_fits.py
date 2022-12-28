

from audioop import avg
from math import nan
import os
import time
import datetime as dt
from binance.helpers import date_to_milliseconds
import numpy as np
from numpy import NaN, dtype, std
from numpy.lib import histograms
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from pandas.core.tools.datetimes import to_datetime
from ta.volatility import average_true_range
from ta import add_all_ta_features
from stored_functions import excel_export
import subprocess
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def excel_export(df, name='temp_file', size=1000000):
    df.head(int(size)).to_excel(str(name) +".xlsx") 
    subprocess.run(["C:/Program Files/Microsoft Office/root/Office16/EXCEL.exe", str(name) +".xlsx"])


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

tf = pd.to_timedelta('5m'.replace('m','min'))
filename = str(os.getcwd())+'\\datasets\\%s-1m-data.csv' % ('LINKUSDT')
raw = pd.read_csv(filename)
raw['timestamp'] = pd.to_datetime(raw['timestamp'])
raw.set_index('timestamp', inplace=True)
raw = raw.resample(tf, label = 'right').last().ffill().apply(pd.to_numeric)

#GET ALL TECHNICAL INDICATOR DATA
raw_ta = add_all_ta_features(raw,open='open',high='high',low='low',close='close', volume='volume')
raw_ta_cut = raw_ta.iloc[45:,]

#FILTER OUT ONLY RELATIVE ONES
cols = ["open","high","low","close","volume","volatility_bbp","volatility_bbhi","volatility_bbli","volatility_kchi","volatility_kcli","volatility_dcl","trend_macd","trend_macd_signal","trend_macd_diff","trend_adx_pos","trend_adx_neg","trend_trix","trend_mass_index","trend_cci","trend_dpo","trend_kst","trend_kst_sig","trend_aroon_up","trend_aroon_down","trend_aroon_ind","trend_psar_up_indicator","trend_psar_down_indicator","momentum_rsi","momentum_stoch_rsi","momentum_stoch_rsi_k","momentum_stoch_rsi_d","momentum_tsi","momentum_uo","momentum_stoch","momentum_stoch_signal","momentum_wr","momentum_ao","momentum_ppo","momentum_ppo_signal","momentum_ppo_hist","others_dr","others_dlr","others_cr"]


#CUT AND PREPROCESS DATASET
raw_ta_cut_t = raw_ta_cut.loc[:,cols]
df = raw_ta_cut_t.apply(pd.to_numeric).copy()
df['returns'] = np.log(df.close.astype(float).div(df.close.astype(float).shift(1)))
#df.returns.describe()
#PLOT MEAN RETURNS
 #df.returns.loc[df['returns']>-0.1,].hist(bins=200)
 #plt.show()
 #df.returns.mean()*1440

df = df['2022']
df = df.iloc[1:,]
df['is_last'] = np.nan
#Iterator of best params tf and pair
#pairs_tfs_combinations = list(itertools.product(pairs,tfs))

#GENERATE PROFITABLE TRADES



#ITERATE STRATEGY TO GET TARGET VALUES
tp = 0.005
sl = -0.003
results = []
position = 0
position_list = []
cum_returns = 0 
cum_returns_list = []
part = []
bar_nr = []
bar_nr_var = 0
trade_nr = []
trade_nr_var = 0


for i in range(len(df)-1):
    if abs(df.returns.iloc[i])> abs(np.percentile(df.returns.to_numpy(), 30)) and position == 0:
        position = np.sign(df.returns.iloc[i])
        position_list.append(position)
        cum_returns = 0#cum_returns + df.returns.iloc[i] * position
        cum_returns_list.append(cum_returns)
        part.append(1)
        bar_nr_var=bar_nr_var+1
        bar_nr.append(bar_nr_var)
        trade_nr_var = trade_nr_var + 1
        trade_nr.append(trade_nr_var)

    elif position != 0 and cum_returns<tp and cum_returns>sl:
        cum_returns = cum_returns + df.returns.iloc[i] * position
        position_list.append(position)
        cum_returns_list.append(cum_returns)
        part.append(2)
        bar_nr_var=bar_nr_var+1
        bar_nr.append(bar_nr_var)
        trade_nr.append(trade_nr_var)

    elif (cum_returns>tp or cum_returns<sl) and position !=0:
        results.append(cum_returns)
        position = 0
        position_list.append(position)
        cum_returns = 0 #cum_returns + df.returns.iloc[i] * position
        cum_returns_list.append(cum_returns)
        part.append(3)
        bar_nr.append(bar_nr_var)
        bar_nr_var = 0
        trade_nr.append(trade_nr_var)
        df['is_last'].iloc[i-1] = 1
        
    elif position == 0:
        position_list.append(0)
        cum_returns_list.append(0)
        part.append(0)
        bar_nr.append(0)
        bar_nr_var=0
        trade_nr.append(np.nan)

print('loop is over')
position_list.append(0)
cum_returns_list.append(0)
part.append(0)
bar_nr.append(0)
trade_nr.append(np.nan)

df['position'] = position_list
df['cum_returns_list'] = cum_returns_list
df['part'] = part
df['bar_nr'] = bar_nr
df['trade_nr'] = trade_nr 



# plt.bar(df.bar_nr.value_counts().iloc[1:30].index,df.bar_nr.value_counts().iloc[1:30])
# plt.show()
#excel_export(df)

#df[['position','cum_returns_list','returns']].head(100)

#CHECK PERFORMANCE
results_df = pd.DataFrame({"cumreturns":results})
results_df['position'] = np.sign(results_df.cumreturns)
results_df['part'] = len(results_df['position'] )
results_df.groupby(['position']).describe()
results_df.groupby(['position']).sum()


#MARK UP TARGETS
only_trades = df[df['is_last']==1]
#excel_export(only_trades, name='only_trades')

#BEST IS TO TAKE TRADES WITH 6-10 BARS
#GENERATING TARGET FOR MODEL
#only_trades['target_tmp'] = np.where((only_trades.cum_returns_list>0) & (only_trades.bar_nr <= 10),1,0)
only_trades['target_tmp'] = np.where(only_trades.cum_returns_list>0,only_trades.position,0)
only_trades.target_tmp.value_counts()

df_main = df.copy()
df_main['timestamp'] = df_main.index
df_main = df_main.merge(only_trades[['target_tmp','trade_nr']], on='trade_nr', how='left', suffixes=('_1', '_2'))
df_main.set_index('timestamp', inplace=True)
df_main['target'] = np.where((df_main.bar_nr==1) & (df_main.target_tmp!=0),df_main.target_tmp,0)


#GENERATING ADDITIONAL PREDICTORS
df_main['time'] = df_main.index
df_main['time'] = pd.to_datetime(df_main['time'])
df_main['dayofweek'] = df_main['time'].dt.day_of_week
df_main['hour'] = df_main['time'].dt.hour
df_main = df_main.drop(['is_last', 'position', 'cum_returns_list', 'part', 'bar_nr', 'trade_nr', 'target_tmp','time'], axis=1)

#excel_export(df_main)
df_main.target.value_counts()

#FIT XGBOOST MODEL
index = df_main.index.to_pydatetime()
start = index[0]
end = index[-1]
setRange = pd.date_range(start, end,freq=dt.timedelta(days=90)).tolist()
dfm = pd.DataFrame()

for i in range(len(setRange)-1):

    if i == len(setRange)-2 :
        dset = df_main.loc[str(setRange[i]):,]
    else:
        dset = df_main.loc[str(setRange[i]):str(setRange[i+1]-pd.Timedelta(minutes=30)),]

    dset['model_subset'] = str(i)
    dfm = dfm.append(dset, ignore_index = False)
print('Splitting on parts is done')

#dfm.model_subset.value_counts()
dfm.target = dfm.target.replace(1,2).replace(-1,1).replace(np.nan,0)
dfm = dfm.fillna(-999).replace(np.inf,-999).replace(-np.inf,-999)
#dfm.to_csv('004_xgboost_model/103_004_xgboost.csv')

## NORMALIZATION

from sklearn.preprocessing import MinMaxScaler
values = dfm.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)
inversed = scaler.inverse_transform(normalized)

## SPLIT ON TRAIN AND TEST

train = dfm.loc[dfm['model_subset'].isin(['0','1','2','3'])]
test = dfm.loc[dfm['model_subset'].isin(['4'])]
X_train = train.drop(['target','model_subset'], axis=1)
y_train = train['target'].astype('int32')
X_test = test.drop(['target','model_subset'], axis=1)
y_test = test['target'].astype('int32')
eval_set = [(X_train, y_train), (X_test, y_test)]

sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=train['target'] #provide your own target name
)

# declare parameters
params = {
            'objective':'multi:softprob',
            'max_depth': 10,
            'min_child_weight':30,
            'num_class': 3,
            'alpha': 0.5,
            'learning_rate':0.5,
            'n_estimators':60
        }         
           
          
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)


# fit the classifier to the training data

xgb_clf.fit(X_train, y_train,eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=False, sample_weight=sample_weights)

y_pred = xgb_clf.predict(X_test)

confusion_matrix(y_test, y_pred)  
xgb.plot_importance(xgb_clf)
plt.show()


result = xgb_clf.evals_result()
epochs = len(result["validation_0"]["merror"])
x_axis = range(0, epochs)


# plot log loss
fig, ax = pyplot.subplots(figsize=(12,12))
ax.plot(x_axis, result["validation_0"]["mlogloss"], label="Train")
ax.plot(x_axis, result["validation_1"]["mlogloss"], label="Test")
ax.legend()
pyplot.ylabel("Log Loss")
pyplot.title("XGBoost Log Loss")
pyplot.show()

# plot classification error
fig, ax = pyplot.subplots(figsize=(12,12))
ax.plot(x_axis, result["validation_0"]["merror"], label="Train")
ax.plot(x_axis, result["validation_1"]["merror"], label="Test")
ax.legend()
pyplot.ylabel("Classification Error")
pyplot.title("XGBoost Classification Error")
pyplot.show()


short = xgb_clf.predict_proba(X_test)[:,1]
long = xgb_clf.predict_proba(X_test)[:,2]

dfc = pd.DataFrame({'y_test':y_test,'y_pred':y_pred,'prob_short':short, 'prob_long':long})
excel_export(dfc)
dfm.to_csv('004_xgboost_model/103_004_xgboost.csv')


#################################################################
# Pipeline
#################################################################
params = {
    'max_depth': 6,
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3,
    'n_gpus': 0
}
pipe_xgb = Pipeline([
    ('clf', xgb.XGBClassifier(**params))
    ])

parameters_xgb = {
        'clf__n_estimators':[30,40], 
        'clf__criterion':['entropy'], 
        'clf__min_samples_split':[15,20], 
        'clf__min_samples_leaf':[3,4]
    }

grid_xgb = GridSearchCV(pipe_xgb,
    param_grid=parameters_xgb,
    scoring='accuracy',
    cv=5,
    refit=True)

#################################################################
# Modeling
#################################################################
start_time = time.time()

grid_xgb.fit(X_train, y_train)

#Calculate the score once and use when needed
acc = grid_xgb.score(X_test,y_test)

print("Best params                        : %s" % grid_xgb.best_params_)
print("Best training data accuracy        : %s" % grid_xgb.best_score_)    
print("Best validation data accuracy (*)  : %s" % acc)
print("Modeling time                      : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

###

#################################################################
# Prediction
#################################################################
#Predict using the test data with selected features
y_pred = grid_xgb.predict(X_test)

# Transform numpy array to dataframe
y_pred = pd.DataFrame(y_pred)

# Rearrange dataframe
y_pred.columns = ['prediction']
y_pred.insert(0, 'id', dfm['id'])
accuracy_score(y_test, y_pred.prediction)
