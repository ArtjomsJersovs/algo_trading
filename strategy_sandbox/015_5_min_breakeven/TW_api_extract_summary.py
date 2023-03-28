from tradingview_ta import TA_Handler, Interval, Exchange, get_multiple_analysis
import pandas as pd
import gspread
import schedule

gc = gspread.service_account('secrets.json')

# gc.del_spreadsheet('1-Q3IjES4jlbTdZIh2llgvl5LTDrvAC4muKWKllxxd5I')
def tradingview_etl_summary():
  analysis = get_multiple_analysis(screener="crypto",interval=Interval.INTERVAL_5_MINUTES,symbols=[ "BINANCE:BTCBUSDPERP","BINANCE:ETHBUSDPERP","BINANCE:DOTBUSDPERP","BINANCE:LINKBUSDPERP","BINANCE:LTCBUSDPERP"])
  df = pd.DataFrame(columns=[
    'ticker', 'time', 'recommend_sum', 'rec_sum_buy', 'rec_sum_sell',
    'rec_sum_neutral', 'rec_osc', 'rec_osc_buy', 'rec_osc_sell',
    'rec_osc_neutral', 'RSI', 'STOCH.K', 'CCI', 'ADX', 'AO', 'Mom', 'MACD',
    'Stoch.RSI', 'WpercentR', 'BBP', 'UO'
  ])

  for x, y in analysis.items():
    ticker = x
    summ = y.summary
    osc = y.oscillators
    osc_compute = y.oscillators['COMPUTE']
    time = y.time

    df = df.append(
      {
        'ticker': ticker,
        'time': str(time),
        'recommend_sum': summ['RECOMMENDATION'],
        'rec_sum_buy': summ['BUY'],
        'rec_sum_sell': summ['SELL'],
        'rec_sum_neutral': summ['NEUTRAL'],
        'rec_osc': osc['RECOMMENDATION'],
        'rec_osc_buy': osc['BUY'],
        'rec_osc_sell': osc['SELL'],
        'rec_osc_neutral': osc['NEUTRAL'],
        'RSI': osc_compute['RSI'],
        'STOCH.K': osc_compute['STOCH.K'],
        'CCI': osc_compute['CCI'],
        'ADX': osc_compute['ADX'],
        'AO': osc_compute['AO'],
        'Mom': osc_compute['Mom'],
        'MACD': osc_compute['MACD'],
        'Stoch.RSI': osc_compute['Stoch.RSI'],
        'WpercentR': osc_compute['W%R'],
        'BBP': osc_compute['BBP'],
        'UO': osc_compute['UO']
      },
      ignore_index=True)

  print(df)

  df_btc = df[(df['ticker']=='BINANCE:BTCBUSDPERP')]
  df_eth =df[(df['ticker']=='BINANCE:ETHBUSDPERP')]
  df_dot = df[(df['ticker']=='BINANCE:DOTBUSDPERP')]
  df_link = df[(df['ticker']=='BINANCE:LINKBUSDPERP')]
  df_ltc = df[(df['ticker']=='BINANCE:LTCBUSDPERP')]

  # sh_5m = gc.create('TW_tech_indicators_summary_5min')
  # sh_5m.share('jersovs.artjoms@gmail.com', perm_type='user', role='writer')
  # worksheet = sh_5m.add_worksheet(title="BTC",rows=90000, cols=21)
  # worksheet = sh_5m.add_worksheet(title="ETH",rows=90000, cols=21)
  # worksheet = sh_5m.add_worksheet(title="DOT",rows=90000, cols=21)
  # worksheet = sh_5m.add_worksheet(title="LINK",rows=90000, cols=21)
  # worksheet = sh_5m.add_worksheet(title="LTC",rows=90000, cols=21)

  sh_5m = gc.open('TW_tech_indicators_summary_5min')
  worksheet_btc =sh_5m.worksheet('BTC')
  worksheet_eth =sh_5m.worksheet('ETH')
  worksheet_dot =sh_5m.worksheet('DOT')
  worksheet_link =sh_5m.worksheet('LINK')
  worksheet_ltc =sh_5m.worksheet('LTC')

  curr_list = [[worksheet_btc,df_btc],[worksheet_eth,df_eth],[worksheet_dot,df_dot],[worksheet_link,df_link],[worksheet_ltc, df_ltc]]

  #run for the first time
  # for ws, df in curr_list:
  #     ws.update([df.columns.values.tolist()] + df.values.tolist())
  #worksheet.update(df.values.tolist())

  #run for incremental 
  cell_to_add_incr = len(worksheet_btc.col_values(1))+1
  range = 'A'+str(cell_to_add_incr)+':'+'U'+str(cell_to_add_incr)

  for ws, df in curr_list:
    ws.update(range, df.values.tolist())
  return(print(f'TW signal parsing successfully finished at: {pd.Timestamp.now()}'))



def scheduled_script():
    
  schedule.every(5).minutes.do(tradingview_etl_summary)
    
  while True:
    schedule.run_pending()

if __name__ == '__main__':
  print('starting_script')
  scheduled_script()