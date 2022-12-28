from codecs import ignore_errors
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import stored_functions as sf
client, bsm = sf.setup_api_conn_binance()
data = sf.get_stored_data_close('BTCBUSD', '1m', '2022-10-01','2022-10-26')
data_2 = sf.get_stored_data_close('BTCUSDT', '1m', '2022-10-01','2022-10-26')
data_2 = data_2.rename({'close':'close2'}, axis=1)

new = pd.concat([data,data_2], axis=1)

new['diff'] = new.close.div(new.close2)-1
new['sign'] = np.sign(new['diff'])
new['diff_abs'] = abs(new['diff'])
new['more_than_comm'] = np.where(new.diff_abs>0.0025,1,0)
new.more_than_comm.value_counts()


plt.hist(new.diff_abs)
plt.show()