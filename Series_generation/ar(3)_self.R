library(forecast)

set.seed(1)

source("./ar_coefficients_generator.R")

lags<-3 #for AR(3) process
maxRoot<-5 # for a non exploding process

# lengths of series
length_ss<-3000

series_per_dataset <- 100
frequency <- 12
no_of_datasets <- 1000
burn_in <- 100

#create the data folder if not existing
#dir.create(file.path(".","2_series"),showWarnings = False)

parameter1<-generate_random_arma_parameters(lags,maxRoot)
parameter2<-generate_random_arma_parameters(lags,maxRoot)

#generating series
full_length_series1 <- NULL
full_length_series2 <- NULL
for (dataset_index in 1:no_of_datasets){
  ts1 <- arima.sim(model=list(ar=parameter1), n=length_ss, n.start = burn_in)
  ts2 <- arima.sim(model=list(ar=parameter2), n=length_ss, n.start = burn_in)
  
  # normalize to 0 mean and unit variance
  ts1 <- scale(ts1)
  ts2 <- scale(ts2)
  min1 <- min(ts1)
  if (min1 < 0){
    ts1 <- ts1 - min1
  }
  if (min1 < 1){
    ts1 <- ts1 + 1
  }
  full_length_series1 <- rbind(full_length_series1, t(ts1))

  min2 <- min(ts2)
  if (min2 < 0){
    ts2 <- ts2 - min2
  }
  if (min2 < 1){
    ts2 <- ts2 + 1
  }
  full_length_series2 <- rbind(full_length_series2, t(ts2))
}

full_length_series1_file <- paste0("./series", dataset_index, ".txt")

write.table(x=full_length_series1, file='./series_1.txt', row.names = FALSE, col.names = FALSE, sep=",")
write.table(x=full_length_series2, file='./series_2.txt', row.names = FALSE, col.names = FALSE, sep=",")
