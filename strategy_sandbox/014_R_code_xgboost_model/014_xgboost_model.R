library(here)
library(future.apply)
library(readxl)
library(excel.link)
library(hmeasure)
library(dplyr)
library(stringr)
library(magrittr)
library(tidyr)
source(here::here("Helper_functions.R"))
library(TTR)
library(xts)

# source(here::here('R','my-internal-projects','system', "load_settings.R"))
#source(here::here(system','R', "xgb wrapper functions summary.R"))

#library(devtools)
library(Sunny)

here()
data_d = read.csv('C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/strategy_sandbox/datasets/BTCBUSD-1d-data.csv')

data_1h = read.csv('C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/strategy_sandbox/datasets/BTCBUSD-1h-data.csv')

snp_data = read.csv('C:/Users/artjoms.jersovs/github/algo_trading/algo_trading/strategy_sandbox/014_R_code_xgboost_model/spy_daily_data.csv')
# feature generation ------------------------------------------------------

data_d = data_d %>% select(-close_time, -quote_av, -tb_base_av, -tb_quote_av, -ignore) %>% 
  mutate(timestamp = as.Date(timestamp),
         close_pct = (close-lag(close))/lag(close) * 100,
         open_pct = (open-lag(open))/lag(open) * 100,
         high_pct = (high-lag(high))/lag(high) * 100,
         low_pct = (low-lag(low))/lag(low) * 100,
         weekday = lubridate::wday(as.Date(data_d$timestamp[1]))) 
         
#indicators
bb <-myBBands(data_d$close,n=20,sd=2)
bb = bb %>% as_tibble() %>% mutate(bb_h_l = up-dn,
                   bb_h_m = up-mavg,
                   bb_m_l = mavg-dn,
                   bb_h_l_pct = (bb_h_l-lag(bb_h_l))/lag(bb_h_l) * 100,
                   bb_h_m_pct = (bb_h_m-lag(bb_h_m))/lag(bb_h_m) * 100,
                   bb_m_l_pct = (bb_m_l-lag(bb_m_l))/lag(bb_m_l) * 100) %>% 
            select(bb_h_l_pct, bb_h_m_pct, bb_m_l_pct, pctB)

# rsi = as_tibble(myRSI(data_d$close, n = 15))
# names(rsi)= c('rsi_15')

ema_15 = as_tibble(myEMA(data_d$close,n = 15))
ema_40 = as_tibble(myEMA(data_d$close,n = 40))
ema = cbind(ema_15,ema_40)
names(ema) = c('ema_15','ema_40')
ema = ema %>% mutate(ema_diff = ema_15-ema_40,
                     ema_diff_pct = (ema_diff-lag(ema_diff))/lag(ema_diff) * 100)

#combine all together
data_d = cbind(data_d, bb, ema)
data_d = data_d %>% select(-open, -low, -high)

snp_data = snp_data %>% mutate(Date = as.Date(snp_data$Date,format = "%m/%d/%y")) %>% filter(Date>=as.Date('2019-09-19'))
names(snp_data) = paste0('snp_',names(snp_data))

snp_data = snp_data %>% 
  mutate(snp_close_pct = (snp_Close-lag(snp_Close))/lag(snp_Close) * 100,
         snp_open_pct = (snp_Open-lag(snp_Open))/lag(snp_Open) * 100,
         snp_high_pct = (snp_High-lag(snp_High))/lag(snp_High) * 100,
         snp_low_pct = (snp_Low-lag(snp_Low))/lag(snp_Low) * 100,
         ) %>% select(-snp_Close, -snp_Open, -snp_High, -snp_Low)

data_d = data_d %>% left_join(snp_data, by=c('timestamp'='snp_Date'))
data_d_proc = data_d
save_rds(data_d_proc)

read_rds(data_d_proc)
# Add lags ----------------------------------------------------------------

lags = 5


for (lag in seq(1,lags)){
  for (col in names(data_d_proc %>% select(close_pct, open_pct, high_pct, low_pct))){
    col_name = paste0(col,'_lag_',lag)
    data_d_proc[[col_name]] = lag(data_d_proc[[col]]) 
  }
}
print('done')

data_d_proc = data_d_proc[-(1:40),]
# train test split --------------------------------------------------------

data_d_proc$random_number <- ceiling(5*runif(nrow(data_d_proc)))

data_d_proc_subsets = data_d_proc %>% mutate(modelling_subset=if_else(random_number==5,'Test','Train'))

#save_rds(app_ids_and_scores_model_subsets)
data_d_proc_subsets$target = case_when(lead(data_d_proc_subsets$close)>data_d_proc_subsets$close~1,T~0)

train_set = data_d_proc_subsets %>% filter(modelling_subset=='Train') %>% rename(response=target)
test_set = data_d_proc_subsets %>% filter(modelling_subset=='Test') %>% rename(response=target)


# fit ---------------------------------------------------------------------

params <- list(
  #nrounds = 1000,
  early_stopping_round = 100,
  eta = 0.005,
  subsample = 0.75, 
  colsample_bytree = 0.9,
  max_depth = 5, #7, 
  min_child_weight = 5 #5
)

model = train_model_get_scored_apps_and_model_metrics(
  train_set_df=train_set,
  hold_out_set_df=test_set,
  features=names(select(train_set, -timestamp, -response, -modelling_subset, -random_number, -close_pct)),
  nround_par=100, #710,
  params=params,
  target_col_name_txt='response', score_col_name='score', exp_pd_col_name='exp_pd')

model$scored_df$exp_bin_result = case_when(model$scored_df$exp_pd>0.545~1,T~0)
model$scored_df$exp_bin_result = case_when(model$scored_df$exp_pd<0.465~-1,T~model$scored_df$exp_bin_result)

X = table(model$scored_df$response,model$scored_df$exp_bin_result)
acc_1 = X[2,2]/(X[1,2]+X[2,2])
acc_0 = X[1,1]/(X[1,1]+X[2,1])
print(X)
print(acc_1)
print(acc_0)





