library(forecast)

set.seed(1)

source("./ar_coefficients_generator.R")

lags <- 3 # for AR(3) process
maxRoot <- 5# for a non exploding process

# lengths of series
length_ss <- 2000

series_per_dataset <- 100
frequency <- 12
no_of_datasets <- 2000
burn_in <- 100

# create the data folder if not existing
dir.create(file.path(".", "original_series"), showWarnings = FALSE)



########### SS and MS-Hom-Short Scenarios
for (dataset_index in 1:no_of_datasets){
  full_length_series <- NULL
  parameters <- generate_random_arma_parameters(lags, maxRoot)
  ts <- arima.sim(model=list(ar=parameters), n=length_ss, n.start = burn_in)
  
  # normalize to 0 mean and unit variance
  ts <- scale(ts)
  min <- min(ts)
  if (min < 0){
    ts <- ts - min
  }
  if (min < 1){
    ts <- ts + 1
  }
  full_length_series <- rbind(full_length_series, t(ts))
  file_name=paste0("original_series/original_series_", dataset_index, ".txt")
  write.table(x=full_length_series, file=file_name, row.names = FALSE, col.names = FALSE, sep=",")
}

