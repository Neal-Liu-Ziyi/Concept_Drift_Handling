from statsforecast.models import AutoARIMA,AutoETS,ARIMA
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from sklearn.metrics import mean_squared_error,mean_absolute_error
from gradual_base_line import series_reader, base_line_model
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

gradual_df=pd.read_csv('../../data/simulated_data/gradual_cd_2000.csv')


auto_arima=AutoARIMA(seasonal=False)
auto_ets=AutoETS()
ar_3=ARIMA(order=[3,0,0])
ar_5=ARIMA(order=[5,0,0])

pred_df,error_df=base_line_model(gradual_df,auto_ets,200)

pred_df.to_csv('../../result/gradual/auto_ets_gradual_window_200_prediction.csv',index=False)
error_df.to_csv('../../result/gradual/auto_ets_gradual_window_200_error.csv',index=False)