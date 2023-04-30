import math
import time

a = 2
b = 3
r = 3

#The lower this value the higher quality the circle is with more points generated
stepSize = 0.08159980918415047

#Generated vertices
positions = []

t = 0
while t < 2 * math.pi:
    positions.append((r * math.cos(t) + a, r * math.sin(t) + b))
    t += stepSize


x, y = zip(*positions)
plt.scatter(*zip(*positions))
plt.show()
len(x)

import pandas as pd
df = pd.DataFrame({'x':x,'y':y})

import stored_functions as sf

sf.excel_export(df)


client, bsm = sf.setup_api_conn_binance()


prices = client.get_all_tickers()

ticker = [i["symbol"] for i in prices]
price = [i["price"] for i in prices]

df = pd.DataFrame(list(zip(ticker, price)),
               columns =['ticker', 'price'])



ethusdt = float(df[df['ticker']=='ETHUSDT']['price'].iloc[0])
atomusdt = float(df[df['ticker']=='ATOMUSDT']['price'].iloc[0])
btcusdt = float(df[df['ticker']=='BTCUSDT']['price'].iloc[0])
ethbtc = ethusdt/btcusdt
atometh = atomusdt/ethusdt
atombtc = atomusdt/btcusdt


bal_btc = 1
n = 0
while n < 100:
    prices = client.get_all_tickers()

    ticker = [i["symbol"] for i in prices]
    price = [i["price"] for i in prices]

    df = pd.DataFrame(list(zip(ticker, price)),
                columns =['ticker', 'price'])

    ethusdt = float(df[df['ticker']=='ETHUSDT']['price'].iloc[0])
    atomusdt = float(df[df['ticker']=='ATOMUSDT']['price'].iloc[0])
    btcusdt = float(df[df['ticker']=='BTCUSDT']['price'].iloc[0])
    ethbtc = ethusdt/btcusdt
    atometh = atomusdt/ethusdt
    atombtc = atomusdt/btcusdt
    
    buy_eth = bal_btc / ethbtc
    buy_atom = buy_eth / atometh
    buy_btc_back = buy_atom * atombtc
    n += 1 
    time.sleep(10)
    print(buy_btc_back)
    




