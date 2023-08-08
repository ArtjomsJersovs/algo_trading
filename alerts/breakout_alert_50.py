import time
import os
import datetime
import statistics
import json
import pandas as pd
from binance import Client
import stored_functions as sf
import telegram_send as ts  #13.5 working version of python-telegram-bot is needed

#ts_conf = '/home/ec2-user/algo_trading/telegram-send.conf'
ts_conf=r'C:\Users\HP\Documents\GitHub\algo_trading\telegram-send.conf'

client = sf.setup_api_conn_binance_only()

# List of futures pairs you want to monitor
futures_pairs = ["CVXUSDT", "GALUSDT", "LDOUSDT", "SOLUSDT", "MATICUSDT", "WAVESUSDT", "SANDUSDT", "BNBUSDT", "NEARUSDT", "GALAUSDT", "UNIUSDT", "PHBUSDT", "TLMUSDT", "AVAXUSDT", "APEUSDT", "LINKUSDT", "FILUSDT", "BTCUSDT", "LTCUSDT", "ETCUSDT", "ADAUSDT", "FTMUSDT", "TRXUSDT", "DOTUSDT", "AMBUSDT", "AGIXUSDT", "APTUSDT", "ETHUSDT", "GMTUSDT", "ANCUSDT"]


output_file = 'C:/Users/HP/Documents/GitHub/algo_trading/futures_data.json'
# output_file = '/home/ec2-user/algo_trading/futures_data.json'
DAYS_TO_MONITOR = 50

# Function to get the 10-day minimum and maximum for a futures pair
def get_x_day_min_max(symbol):
    klines = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=DAYS_TO_MONITOR)
    closes = [float(kline[4]) for kline in klines]
    min_price = min(closes)
    max_price = max(closes)
    current_price = closes[-1]
    return min_price, max_price, current_price 

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
            min_price, max_price, current_price = get_x_day_min_max(pair)

            current_time = datetime.datetime.now()
            last_alert_time =  datetime.datetime.strptime(cooldown_dict[pair], "%Y-%m-%d %H:%M:%S.%f")
            hours_since_last_alert = (current_time - last_alert_time).total_seconds()/3600

            if hours_since_last_alert>=3:
                if 1-(current_price/max_price)<=0.001:
                    message = f"ALERT: {pair} price ({current_price}) is 0.1% close to 50-day maximum ({max_price})."
                    ts.send(conf=ts_conf, messages=[message])
                    cooldown_dict[pair] = str(datetime.datetime.now())
                    with open(output_file, 'w') as json_file:
                        json.dump(cooldown_dict, json_file, indent=4)  

                if 1-(min_price/current_price)<=0.001:
                    message = f"ALERT: {pair} price ({current_price}) is 0.1% close to 50-day minimum ({min_price})."
                    ts.send(conf=ts_conf, messages=[message])
                    cooldown_dict[pair] = str(datetime.datetime.now())
                    with open(output_file, 'w') as json_file:
                        json.dump(cooldown_dict, json_file, indent=4) 

    except Exception as e:
        print(f"Error occurred: {e}")
        ts.send(conf=ts_conf, messages=['Alert error: '+e])

if __name__ == "__main__":
    main()