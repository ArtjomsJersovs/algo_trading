import time
import os
import datetime
import statistics
import json
import numpy as np
import pandas as pd
from binance import Client
import stored_functions as sf
import telegram_send as ts  #13.5 working version of python-telegram-bot is needed

#ts_conf = '/home/ec2-user/algo_trading/telegram-send.conf'
ts_conf=r'C:\Users\HP\Documents\GitHub\algo_trading\telegram-send.conf'

client = sf.setup_api_conn_binance_only()

# List of futures pairs you want to monitor
futures_pairs = ["CVXUSDT", "GALUSDT", "HIGHUSDT","LDOUSDT", "SOLUSDT", "MATICUSDT", "WAVESUSDT", "SANDUSDT", "BNBUSDT", "NEARUSDT", "GALAUSDT", "UNIUSDT", "PHBUSDT", "TLMUSDT", "AVAXUSDT", "APEUSDT", "LINKUSDT", "FILUSDT", "LTCUSDT", "ETCUSDT", "ADAUSDT", "FTMUSDT", "TRXUSDT", "DOTUSDT", "AMBUSDT", "AGIXUSDT", "APTUSDT", "ETHUSDT", "GMTUSDT"]


output_file = 'C:/Users/HP/Documents/GitHub/algo_trading/futures_data.json'
# output_file = '/home/ec2-user/algo_trading/futures_data.json'

# Function to get the 10-day minimum and maximum for a futures pair
def get_corr_and_vsa(symbol1, symbol2='BTCUSDT'):
    klines1 = client.futures_klines(symbol=symbol1, interval=Client.KLINE_INTERVAL_5MINUTE, limit=100)
    klines2 = client.futures_klines(symbol=symbol2, interval=Client.KLINE_INTERVAL_5MINUTE, limit=12)
    closes1 = [float(kline[4]) for kline in klines1[-12:]]
    closes2 = [float(kline[4]) for kline in klines2]

    #calculate correlation with BTC
    corr_coef = np.corrcoef(closes1, closes2)[0, 1]

    ##calculate average volume
    volumes = [float(kline[5]) for kline in klines1]
    df = pd.DataFrame(dict(volume=volumes))
    df['ma'] = df['volume'].rolling(15).mean()
    df['rel_vol'] = df.volume/df.ma
    rel_vol = df['rel_vol'].iloc[-1]
    return corr_coef, rel_vol

# Main function for monitoring and sending alerts
def main():
        #Read main trade parameters
    if os.path.isfile(output_file):
        with open(output_file, 'r') as json_file:
            cooldown_dict = json.load(json_file)
    else:
        cooldown_dict = {pair: str(datetime.datetime.now()- datetime.timedelta(days=1)) for pair in futures_pairs}
        with open(output_file, 'w') as json_file:
            json.dump(cooldown_dict, json_file, indent=4)

    try:
        for pair in futures_pairs:
            corr, rel_vol = get_corr_and_vsa(pair)

            current_time = datetime.datetime.now()
            last_alert_time =  datetime.datetime.strptime(cooldown_dict[pair], "%Y-%m-%d %H:%M:%S.%f")
            hours_since_last_alert = (current_time - last_alert_time).total_seconds()/3600

            if hours_since_last_alert>=3:
                if corr<0.6 and rel_vol>4:
                    message = f"ALERT: {pair} Correlation to BTC: ({str(round(corr,2))}) and relative volume: ({str(round(rel_vol,2))})."
                    ts.send(conf=ts_conf, messages=[message])
                    cooldown_dict[pair] = str(datetime.datetime.now())
                    with open(output_file, 'w') as json_file:
                        json.dump(cooldown_dict, json_file, indent=4)  

    except Exception as e:
        print(f"Error occurred: {e}")
        ts.send(conf=ts_conf, messages=['Alert error: '+e])

if __name__ == "__main__":
    main()



