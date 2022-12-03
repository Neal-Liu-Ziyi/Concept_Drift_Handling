import pandas as pd
import itertools
import numpy as np
import random
from random import choices
from pandas.plotting import autocorrelation_plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import combinations
from sklearn.metrics import mean_absolute_percentage_error


# Function define

def RSS_calculator(y, y_hat):
    RSS = np.sum((y - y_hat) ** 2)
    return RSS


def string_to_list(string_org):
    string = string_org.split(',')
    string_list = []
    for i in range(len(string)):
        if i == 0:
            data = float(string[i][1:])

        elif i == len(string) - 1:
            data = float(string[i][:-1])
        else:
            data = float(string[i])
        string_list.append(data)
    return string_list


def RSS_Based_Gradient(y, y_hat1, y_hat2, weight_1, weight_2):
    error = (y - weight_1 * y_hat1 - weight_2 * y_hat2)
    grad1 = -2 * y_hat1 * error
    grad2 = -2 * y_hat2 * error

    return grad1, grad2


def df_pair(temp_list, n):
    temp_list2 = []
    for c in combinations(temp_list, n):
        temp_list2.append(c)
    return temp_list2


window_size_list = [1]

window_size_list = [1]


def ECW_method(error_function, window_size_list, drfit_type_df, df1, df2):
    """
    error_functio: RSS,...

    drfit_type_df: sudden_df,gradual_df,incremental_df

    """

    window_size_df_RMSE = pd.DataFrame({
        'window_size_1_RMSE': []
    })

    window_size_df_MASE = pd.DataFrame({
        'window_size_1_MASE': []
    })
    for row in range(len(drfit_type_df)):

        series_row = drfit_type_df.loc[row]
        series = series_row['series']
        series_list = string_to_list(series)

        df1_predic_df_perdiction = df1.iloc[:, row]
        df2__predic_df_perdiction = df2.iloc[:, row]

        weighted_df = pd.DataFrame({
            'df1_predic': df1_predic_df_perdiction,
            'df2_predic': df2__predic_df_perdiction,
            'label': series_list[-len(df1):]
        })
        weighted_df['weighted_perdiction'] = np.NaN
        RMSE_list = []
        MAE_list = []
        for x in range(len(window_size_list)):
            for i in range(len(weighted_df)):
                if i < window_size_list[x]:
                    weighted_df.iloc[i, 3] = 0.5 * weighted_df.iloc[
                        i, 0] + 0.5 * weighted_df.iloc[i, 1]

                else:
                    df1_error_sum = error_function(
                        weighted_df.iloc[i - window_size_list[x]:i, 2],
                        weighted_df.iloc[i - window_size_list[x]:i, 0])

                    df2_error_sum = error_function(
                        weighted_df.iloc[i - window_size_list[x]:i, 2],
                        weighted_df.iloc[i - window_size_list[x]:i, 1])
                    sum_error = df1_error_sum + df2_error_sum
                    df1_error_percentage = df1_error_sum / sum_error

                    weighted_df.iloc[
                        i, 3] = (1 - df1_error_percentage) * weighted_df.iloc[
                        i, 0] + df1_error_percentage * weighted_df.iloc[i,
                                                                        1]

            window_weighted_RMSE = np.sqrt(
                ((weighted_df['weighted_perdiction'] -
                  weighted_df['label']) ** 2).mean())
            window_weighted_MASE = mean_absolute_error(
                weighted_df['weighted_perdiction'], weighted_df['label'])

            RMSE_list.append(window_weighted_RMSE)
            MASE_list.append(window_weighted_MASE)

        window_size_df_RMSE.loc[row] = RMSE_list
        window_size_df_MAE.loc[row] = MAE_list
    return window_size_df_RMSE, window_size_df_MAE


def GDW_method(error_function,
               window_size_list,
               drfit_type_df,
               df1,
               df2,
               learning_rate=0.01):
    """
    error_functio: RSS,...

    drfit_type_df: sudden_df,gradual_df,incremental_df

    """

    window_size_df_RMSE = pd.DataFrame({
        'window_size_1_RMSE': [],
        'window_size_2_RMSE': [],
        'window_size_3_RMSE': [],
        'window_size_5_RMSE': [],
        'window_size_10_RMSE': [],
        'window_size_15_RMSE': []
    })

    for row in range(len(drfit_type_df)):

        series_row = drfit_type_df.loc[row]
        series = series_row['series']
        series_list = string_to_list(series)

        df1_predic_df_perdiction = df1.iloc[:, row]
        df2__predic_df_perdiction = df2.iloc[:, row]

        weighted_df = pd.DataFrame({
            'df1_predic': df1_predic_df_perdiction,
            'df2_predic': df2__predic_df_perdiction,
            'label': series_list[-len(df1):]
        })
        weighted_df['weighted_perdiction'] = np.NaN
        RMSE_list = []
        MAE_list = []
        weight_1 = 0.5
        weight_2 = 0.5
        for x in range(len(window_size_list)):
            for i in range(len(weighted_df)):
                if i < window_size_list[x]:
                    weighted_df.iloc[i, 3] = 0.5 * weighted_df.iloc[
                        i, 0] + 0.5 * weighted_df.iloc[i, 1]

                else:
                    y = weighted_df.iloc[i - window_size_list[x]:i,
                        2].mean(axis=0)
                    y_hat1 = weighted_df.iloc[i - window_size_list[x]:i,
                             0].mean(axis=0)
                    y_hat2 = weighted_df.iloc[i - window_size_list[x]:i,
                             1].mean(axis=0)

                    grad_1, grad_2 = error_function(y=y,
                                                    y_hat1=y_hat1,
                                                    y_hat2=y_hat2,
                                                    weight_1=weight_1,
                                                    weight_2=weight_2)

                    weight_1_update = weight_1 - grad_1 * learning_rate
                    weight_2_update = weight_2 - grad_2 * learning_rate
                    weighted_df.iloc[
                        i, 3] = weight_1_update * weighted_df.iloc[
                        i, 0] + weight_2_update * weighted_df.iloc[i, 1]
                    weight_1 = weight_1_update
                    weight_2 = weight_2_update

            window_weighted_RMSE = np.sqrt(
                ((weighted_df['weighted_perdiction'] -
                  weighted_df['label']) ** 2).mean())
            window_weighted_MASE = mean_absolute_error(
                weighted_df['weighted_perdiction'], weighted_df['label'])

            RMSE_list.append(window_weighted_RMSE)
            MAE_list.append(window_weighted_MASE)

        window_size_df_RMSE.loc[row] = RMSE_list
        window_size_df_MAE.loc[row] = MAE_list

    return window_size_df_RMSE, window_size_df_MAE