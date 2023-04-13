library(here)
library(future.apply)
library(readxl)
library(excel.link)
library(hmeasure)
library(dplyr)
library(stringr)
library(magrittr)
library(tidyr)
library(Sunny)
# library(reticulate)
# py_install("numpy")

data = read_excel('C:\\Users\\artjoms.jersovs\\github\\algo_trading\\algo_trading\\strategy_sandbox\\016_trading_view_ta_signals\\tw_data_clean.xlsx')
names = colnames(data)
names[1] = 'timestamp'
colnames(data) = names
test = Feature_Information_Value(
  data,
  data %>% select(-close,-open,-high,-low,-timestamp,-volume,-returns,-ticker) %>% names(),
  response.var='returns',
  p.margin = 0.10,
  excel.var.export = TRUE,
  pdf.var.export = FALSE,
  base_plots = TRUE,
  output_formatting = T
)
