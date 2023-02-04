
# Select libraries  -------------------------------------------------------


Sys.setenv(TZ="GMT")
library(rlist)
library(here)
library(tibble)
library(stringr)
library(data.table)
#library(dtplyr)
library(gtools)



##########################################################
# 
# Get variable importance, holdout distribution, badrate vs predictions (transformed to logodds)
#
###########################################################

get_model_summary <-function (model, holdout_df, bin_size=50, use_identity_position_type = T) 
{
  
  importanceGen<-xgb.importance(feature_names = colnames(data.matrix(select(holdout_df,-response))),
                                model = model)
  # xgb.plot.multi.trees(names(select(data.train.xgb.simplified.3,-response)), model = model)
  
  # Results on holdout for the 3rd model iteration
  predictions.on.holdout<-predict(model,
                                  data.matrix(select(subset(holdout_df,response %in% c(0,1)),-response))
  )
  
  results <- hmeasure::HMeasure(subset(holdout_df,response %in% c(0,1))$response,predictions.on.holdout)
  
  
  #****************************************************************
  # Plot Holdout good/bad distribution by score
  holdout_set <- cbind((subset(holdout_df,response %in% c(0,1))),predictions.on.holdout) %>% 
    mutate(score = Sunny::getScore(predictions.on.holdout))
  
  holdout_pred_log_vs_bdrt <- holdout_set %>% 
    group_by(score_gr = ceiling(score/bin_size)*bin_size) %>% 
    summarise(count = n(),
              count_prc = n()/ nrow(.),
              mean_br = mean(response, na.rm = T),
              mean_predictions.on.modelling.simplified = mean(predictions.on.holdout, na.rm = T)) %>% 
    mutate(
      dplyr::na_if(log((1-mean_br)/mean_br), Inf),
      mean_predictions.on.modelling.simplified_log_odds = log((1-mean_predictions.on.modelling.simplified)/mean_predictions.on.modelling.simplified),
      mean_br_log_odds = log((1-mean_br)/mean_br)
    ) 
  
  log_vs_bdrt_plot <- ggplot2::ggplot(holdout_pred_log_vs_bdrt, aes(x=score_gr, y=mean_br_log_odds, color = 'Badrate'))+
    geom_line(color = 'red', size = 1.2)+
    geom_line(aes(y=mean_predictions.on.modelling.simplified_log_odds, color = 'Predictions' ), color = 'darkgreen',size = 1.2)+
    geom_smooth(method='lm', se = F, size = 1.2, alpha=0.8, color = 'darkorange')+
    theme_minimal()
  
  
  

  
  distribution.g.b. <- final.plots.plot.distribution.g.b.(
    holdout_set,
    score_name = "score",
    response_name = "response",
    plot.segment.txt = "Good/ Bad",
    bin_size = 50,
    use_identity_position_type = T
  )
  
  

  training_auc_plot <-model[['evaluation_log']] %>%
                      ggplot2::ggplot(aes(x=iter, y = train_auc))+
                      geom_line(size = 1.5, color = 'darkblue')+
                      xlab('Iterations')+
                      ylab('Training AUC')+
                      theme_minimal()
  
  
  # Export model performance, variable importance and good/bad distribution on holdout
  
  xl.workbook.add()
  xl.sheet.add('Model_summary', before = 'Sheet1')
  xlc[b1]  <- results[["metrics"]]
  xlc[a5]  <- importanceGen
  plot(distribution.g.b.)
  xlrc[g5] <- current.graphics()
  xl.sheet.add('Bdrt_vs_predictions', before = 'Sheet1')
  xlc[a1]<-holdout_pred_log_vs_bdrt
  plot(log_vs_bdrt_plot)
  xlrc[j5] <-current.graphics()
  xl.sheet.add('Params', before = 'Sheet1')
  plot(training_auc_plot)
  xlrc[j5] <-current.graphics()
  
  
}


export_grid_results_to_excel=function(grid_results){
  xl.workbook.add()
  xl.sheet.add('summary_frame'); xlc[a1] = grid_results$summary_frame
  if(!is.null(grid_results$Variable_analysis$feature_subset_number_detection_df)) {
    xl.sheet.add('feat_subset_numb_detect_df'); xlc[a1] =grid_results$Variable_analysis$feature_subset_number_detection_df
  }
  xl.sheet.add('var_exclusion_effect'); xlc[a1] = grid_results$Variable_analysis$var_exclusion_effect
  xl.sheet.add('top_runs_vars'); xlc[a1] = grid_results$Variable_analysis$top.runs.vars
  xl.sheet.delete('Sheet1')
}



remove_duplicates <- function(dataframe, by_column){
  require(dplyr)
  by_column <-enquo(by_column)
  dataframe %>% group_by(!!by_column) %>% mutate(rownum = row_number()) %>% 
    filter(rownum==1) %>% select(-rownum) %>% ungroup()
}


yearmonth <- function (date){
  as.numeric(format(as.Date(date), "%Y%m"))
}



add_prefix_to_cols_in_df <- function(df_in, prefix_text){
  df_out <- df_in
  colnames(df_out) <- paste0(prefix_text, colnames(df_in))
  return(df_out)
}


excel_export <- function (data){
  require(excel.link)
  xl.workbook.add()
  xlc[a1]<- data
}


right = function (string, char) {
  substr(string,nchar(string)-(char-1),nchar(string))
}

left = function (string,char) {
  substr(string,1,char)
}

SMA <- function (price,n){
  sma <- c()
  sma[1:(n-1)] <- NA
  for (i in n:length(price)){
    sma[i]<-mean(price[(i-n+1):i])
  }
  sma <- reclass(sma,price)
  return(sma)
}


myBBands <- function (price,n,sd){
  mavg <- SMA(price,n)
  sdev <- rep(0,n)
  N <- length(price)
  for (i in (n+1):N){
    sdev[i]<- sd(price[(i-n+1):i])
  }
  up <- mavg + sd*sdev
  dn <- mavg - sd*sdev
  pctB <- (price - dn)/(up - dn)
  output <- cbind(dn, mavg, up, pctB)
  colnames(output) <- c("dn", "mavg", "up", 
                        "pctB")
  return(output)
}

myEMA <- function (price,n){
  ema <- c()
  ema[1:(n-1)] <- NA
  ema[n]<- mean(price[1:n])
  beta <- 2/(n+1)
  for (i in (n+1):length(price)){
    ema[i]<-beta * price[i] + 
      (1-beta) * ema[i-1]
  }
  #ema <- reclass(ema,price)
  return(ema)
}


myRSI <- function (price,n){
  N <- length(price)
  U <- rep(0,N)
  D <- rep(0,N)
  rsi <- rep(NA,N)
  Lprice <- lag(price,1)
  for (i in 2:N){
    if (price[i]>=Lprice[i]){
      U[i] <- 1
    } else {
      D[i] <- 1
    }
    if (i>n){
      AvgUp <- mean(U[(i-n+1):i])
      AvgDn <- mean(D[(i-n+1):i])
      rsi[i] <- AvgUp/(AvgUp+AvgDn)*100 
    }
  }
  rsi <- reclass(rsi, price)
  return(rsi)
}
