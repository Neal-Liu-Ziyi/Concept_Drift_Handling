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


def light_gbm_para_generator(num_lr,num_sub_feature,Num_leaves,num_min_data,num_max_depth):

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

    light_gbm_para_dic={
        'LR_list':LR_list,
        'sub_feature_list':sub_feature_list,
        'min_data_list':min_data_list,
        'max_depth_list':max_depth_list,
        'boost_type_list':boost_type_list
    }

    return light_gbm_para_dic


def prequential_CV(K, training_set,parameter_dict):
    ave_trainig_RMSE = 0
    ave_training_MAE = float('inf')

    #for dic_key in parameter_dict.keys():
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
                                for i in range(1, len(alpha)):
                                    alpha[i] = alpha[i - 1] * alpha[0]
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