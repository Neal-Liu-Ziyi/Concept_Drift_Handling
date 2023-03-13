from statsforecast.models import AutoARIMA,AutoETS,ARIMA
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from sklearn.metrics import mean_squared_error,mean_absolute_error
from incremental_base_line_model import incremental_series_reader, incremental_base_line_model

incremental_df=pd.read_csv('../../data/simulated_data/incremental_cd_2000.csv')
incremental_df=incremental_df.sort_values('start_point',ascending=True)
incremental_df.reset_index(drop=True, inplace=True)

auto_arima=AutoARIMA(seasonal=False)
auto_ets=AutoETS()
ar_3=ARIMA(order=[3,0,0])
ar_5=ARIMA(order=[5,0,0])

pred_df,error_df=incremental_base_line_model(incremental_df,auto_ets,'all')

pred_df.to_csv('../../result/incremental/auto_ets_incremental_window_all_prediction.csv',index=False)
error_df.to_csv('../../result/incremental/auto_ets_incremental_window_all_error.csv',index=False)