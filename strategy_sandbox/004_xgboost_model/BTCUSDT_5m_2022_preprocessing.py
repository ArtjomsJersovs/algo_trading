from csv import excel
from math import nan
import os
import time
import datetime as dt
import numpy as np
from numpy.lib import histograms
import pandas as pd
from stored_functions import excel_export
from pandas.core.tools.datetimes import to_datetime
from stored_functions import excel_export
import subprocess
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.style.use("ggplot")


df = pd.read_csv('C:/Users/artjoms.jersovs/github/AJbots/strategy_sandbox/004_xgboost_model/BTCUSDT_5m_2022_proc.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

#NORMALIZE PREDICTORS
values = df.values#df.drop(['close','returns'], axis=1).values
values = values.reshape((len(values), 46))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
df_norm = scaler.transform(values)
#inversed = scaler.inverse_transform(normalized)

df = pd.DataFrame(data=df_norm, columns=df.columns)

df.returns.hist(bins=200)
plt.show()


#DIMENSION REDUCTION WITH PCA
pca = PCA(n_components=3)
dim_red_feat = pca.fit_transform(df_norm, y=None)
df['PCA1'] = dim_red_feat[:,0]
df['PCA2'] = dim_red_feat[:,1]
df['PCA3'] = dim_red_feat[:,2]
df = df.loc[:,['returns','close','PCA1','PCA2','PCA3']]

plt.scatter(df['2022'].returns, df.loc['2022-06'].momentum_rsi, c ="pink",
            linewidths = 2,
            marker ="s",
            edgecolor ="green",
            s = 50)
plt.show()




#AUTO CORRELATION
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(df["close"])
# plt.show()

# from statsmodels.tsa.seasonal import seasonal_decompose
# res = seasonal_decompose(df['returns'], model = "additive",period = 10)

# fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,8))
# res.trend.plot(ax=ax1,ylabel = "trend")
# res.resid.plot(ax=ax2,ylabel = "seasoanlity")
# res.seasonal.plot(ax=ax3,ylabel = "residual")
# plt.show()
