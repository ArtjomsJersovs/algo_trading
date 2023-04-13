import pandas as pd
import datetime
import stored_functions as sf
import numpy as np

tw_data = pd.read_excel(r'strategy_sandbox\016_trading_view_ta_signals\tw_data.xlsx')

tw_data['time'] = pd.Series(pd.to_datetime(tw_data['time'],  format = "%Y-%m-%d %H:%M:%S"))
tw_data['time'] = tw_data['time'].dt.floor('5min')
tw_data = tw_data.set_index('time')

data_5m = sf.get_stored_data_close('BTCBUSD','5m',"2023-03-28","2023-04-04")
data_5m_tw = tw_data.join(data_5m)

#sf.excel_export(data_5m_tw)

#returns
data_5m_tw['returns'] = np.sign(np.log(data_5m_tw.close.astype(float).div(data_5m_tw.close.astype(float).shift(1)))).shift(-1)

