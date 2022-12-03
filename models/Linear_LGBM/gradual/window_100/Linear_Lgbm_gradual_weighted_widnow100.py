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

gradual_df=pd.read_csv('../gradual_cd_2000.csv')

def string_to_list(string_org):
    string=string_org.split(',')
    string_list=[]
    for i in range(len(string)):
        if i==0:
            data=float(string[i][1:])

        elif i==len(string)-1:
            data=float(string[i][:-1])
        else:
            data=float(string[i])
        string_list.append(data)
    return string_list

def para_num(num_lr,num_sub_feature,Num_leaves,num_min_data,num_max_depth):
    LR_list=[np.random.uniform(0, 1) for i in range(num_lr) ]
    LR_list.sort()


    sub_feature_list=[np.random.uniform(0, 1) for i in range(num_sub_feature)]
    sub_feature_list.sort()

    num_leaves_list=[np.random.randint(20, 300) for i in range(Num_leaves)]
    num_leaves_list.sort()

    min_data_list=[np.random.randint(10, 100) for i in range(num_min_data)]
    min_data_list.sort()


    max_depth_list=[np.random.randint(50, 300) for i in range(num_max_depth)]
    max_depth_list.sort()
    boost_type_list=['goss']
    return LR_list,sub_feature_list,num_leaves_list,min_data_list,max_depth_list,boost_type_list


# define a prequential method
def prequential_CV(K, training_set):
    ave_trainig_RMSE = 0
    ave_training_MAE = float('inf')
    for learning_rate in LR_list:
        for boosting_type in boost_type_list:
            for sub_feature in sub_feature_list:
                for num_leaves in num_leaves_list:
                    for min_data in min_data_list:
                        for max_depth in max_depth_list:
                            params = {'learning_rate': learning_rate,
                                      'boosting_type': boosting_type,
                                      'sub_feature': sub_feature,
                                      'num_leaves': num_leaves,
                                      'min_data': min_data,
                                      'max_depth': max_depth,
                                      'verbosity': -1,
                                      'feature_pre_filter': False}

                            valid_pre_list = list()
                            rows = training_set.count()[0]
                            sum_metric = 0
                            for k in range(K):
                                if k == K - 1:
                                    break
                                train = training_set.iloc[:round((k + 1) * rows / K)]
                                valid = training_set.iloc[round((k + 1) * rows / K):round((k + 2) * rows / K)]
                                train_x = train.drop(columns=['value'], axis=1)
                                train_y = train['value']

                                valid_lable = valid['value']
                                valid_x = valid.drop(columns=['value'], axis=1)

                                alpha = np.zeros(len(train_x))
                                alpha[0] = 0.9
                                beta = 0.9 / len(train_x)
                                for i in range(1, len(alpha)):
                                    alpha[i] = alpha[i - 1] - beta
                                alpha = np.flipud(alpha)
                                train_set = lgb.Dataset(train_x, label=train_y, weight=alpha)

                                model_gbm = lgb.train(params,
                                                      train_set)

                                valid_pre = model_gbm.predict(valid_x)
                                valid_pre_list.append(valid_pre)

                            all_predic = np.array(list(itertools.chain.from_iterable(valid_pre_list)))
                            valid_label = training_set.iloc[round((1) * rows / K):]['value']
                            RMSE = np.sqrt(((all_predic - valid_label) ** 2).mean())
                            MAE = mean_absolute_error(all_predic, valid_label)

                            if MAE < ave_training_MAE:
                                best_params_dic = {
                                    'learning_rate': learning_rate,
                                    'boosting_type': boosting_type,
                                    'sub_feature': sub_feature,
                                    'num_leaves': num_leaves,
                                    'min_data': min_data,
                                    'max_depth': max_depth}
                                ave_trainig_RMSE = RMSE
                                ave_training_MAE = MAE

    return ave_trainig_RMSE, ave_training_MAE, best_params_dic


predictions_df = pd.DataFrame()
prediction_mean_df = pd.DataFrame()
training_mean_rmse_list = []
training_mean_mae_list = []
testing_mean_rmse_list = []
testing_mean_mae_list = []
for row in range(len(gradual_df)):
    rmse_list = []
    MAE_list = []
    training_rmse_list = []
    training_mae_list = []

    gradual_series_row = gradual_df.loc[row]
    gradual_series = gradual_series_row[1]
    gradual_series_list = string_to_list(gradual_series)
    gradual_series_df = pd.DataFrame({})
    gradual_series_df = pd.DataFrame({
        'value': gradual_series_list})
    for lags in range(1, 11):
        gradual_series_df['lag' + str(lags)] = gradual_series_df['value'].shift(lags).fillna(0)

    # gradual_series_df['window_std']=np.std(gradual_series_df.loc[:,
    #                       (gradual_series_df.columns!='value')&(gradual_series_df.columns!='drift_point')],
    #                                ddof=1,axis=1)

    training_set_all = gradual_series_df.iloc[:round(0.8 * len(gradual_series_df))]
    test_set = gradual_series_df.iloc[round(0.8 * len(gradual_series_df)):]

    # trainign with fixed window
    training_set = training_set_all.iloc[-100:]

    train_x = training_set.drop(columns=['value'], axis=1)
    train_y = training_set['value']

    alpha_testing = np.zeros(len(train_x))
    alpha_testing[0] = 0.9
    beta_testing = 0.9 / len(train_x)
    for i in range(1, len(alpha_testing)):
        alpha_testing[i] = alpha_testing[i - 1] - beta_testing
    alpha_testing = np.flipud(alpha_testing)
    train_set = lgb.Dataset(train_x, label=train_y, weight=alpha_testing)

    test_x = test_set.drop(columns=['value'], axis=1)
    test_y = test_set['value']

    for times in range(100):
        LR_list, sub_feature_list, num_leaves_list, min_data_list, max_depth_list, boost_type_list = para_num(1, 1, 1,
                                                                                                              1, 1)

        ave_trainig_RMSE, ave_training_MAE, best_params_dic = prequential_CV(8, training_set=training_set)

        params = {'learning_rate': best_params_dic['learning_rate'],
                  'boosting_type': best_params_dic['boosting_type'],
                  'sub_feature': best_params_dic['sub_feature'],
                  'num_leaves': best_params_dic['num_leaves'],
                  'min_data': best_params_dic['min_data'],
                  'max_depth': best_params_dic['max_depth'],
                  'verbosity': -1,
                  'feature_pre_filter': False}

        best_model = lgb.train(params, train_set)
        test_predict = best_model.predict(test_x)

        testing_RMSE = np.sqrt(((test_predict - test_y) ** 2).mean())
        testing_MAE = mean_absolute_error(test_predict, test_y)

        rmse_list.append(testing_RMSE)
        MAE_list.append(testing_MAE)
        training_rmse_list.append(ave_trainig_RMSE)
        training_mae_list.append(ave_training_MAE)
        predictions_df['times' + str(times)] = test_predict

    predictions = predictions_df.mean(axis=1)

    testing_rmse_array = np.array(rmse_list)
    testing_mae_array = np.array(MAE_list)
    training_rmse_array = np.array(training_rmse_list)
    training_mae_array = np.array(training_mae_list)

    testing_rmse_mean = np.mean(testing_rmse_array)
    testing_mae_mean = np.mean(testing_mae_array)
    training_rmse_mean = np.mean(training_rmse_array)
    training_mae_mean = np.mean(training_mae_array)

    testing_mean_rmse_list.append(testing_rmse_mean)
    training_mean_mae_list.append(testing_mae_mean)
    training_mean_rmse_list.append(training_rmse_mean)
    testing_mean_mae_list.append(training_mae_mean)

    prediction_mean_df['alpha_index' + str(row)] = predictions

gradual_df['testing_RMSE'] = testing_mean_rmse_list
gradual_df['testing_MAE'] = training_mean_mae_list
gradual_df['training_RMSE'] = training_mean_rmse_list
gradual_df['training_MAE'] = testing_mean_mae_list


gradual_df.to_csv('Linear_gradual_cd_weighted_cv8_window100.csv',index=False)
prediction_mean_df.to_csv('Linear_gradual_prediction_weighted_cv8_window100.csv',index=False)