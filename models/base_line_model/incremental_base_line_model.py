from statsforecast.models import AutoARIMA,AutoETS,ARIMA
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from sklearn.metrics import mean_squared_error,mean_absolute_error


def incremental_series_reader(df,row):
    series=df.iloc[row,2]
    # remove the brackets and split the string into individual elements
    string_arr = series.replace('[','').replace(']','').split(',')

    # convert each string element to float and create a numpy array
    arr = np.array([float(x) for x in string_arr])
    return arr

def incremental_base_line_model(df,model,window_size):
    error_df=pd.DataFrame({'start_point':[],
                           'ending_point':[],
                           'testing_RMSE':[],
                           'testing_MAE':[]})
    
    pred_df=pd.DataFrame()
    for row in range(len(df)):
        starting_point = df.iloc[row, 0]
        ending_poing = df.iloc[row, 1]
        all_series=incremental_series_reader(df,row)

        pred_list=[]
        if type(window_size) is int:
            training_series=all_series[:round(0.8 * len(all_series))]
            testing_series=all_series[round(0.8 * len(all_series)):]
            training_series=training_series[int(window_size)*(-1):]
            meaningful_series=np.append(training_series,testing_series)
            for i in range(len(meaningful_series)):
                if i<len(meaningful_series)-window_size:
                    training_set=meaningful_series[i:i+window_size]
                    pred=model.fit(training_set).predict(1)['mean'][0]
                    pred_list.append(pred)
            rmse=mean_squared_error(squared=False,y_true=testing_series,y_pred=pred_list)
            mae=mean_absolute_error(testing_series,pred_list)


        elif window_size=='all':
            training_series=all_series[:round(0.8 * len(all_series))]
            testing_series=all_series[round(0.8 * len(all_series)):]
            for i in range(len(testing_series)):
                training_set=np.append(training_series,testing_series[:i])
                pred=model.fit(training_set).predict(1)['mean'][0]
                pred_list.append(pred)


        rmse=mean_squared_error(squared=False,y_true=testing_series,y_pred=pred_list)
        mae=mean_absolute_error(testing_series,pred_list)

        pred_df['drift_point' + str(starting_point) + 'ending_point' + str(ending_poing)]=pred_list
        error_df.loc[row,:]=[starting_point,ending_poing,rmse,mae]
    return pred_df,error_df