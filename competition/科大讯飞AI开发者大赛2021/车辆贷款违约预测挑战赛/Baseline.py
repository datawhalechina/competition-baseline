#!/usr/bin/env python
# coding: utf-8


# 导入第三方包
import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')



# 读取数据集，具体下载方式可见操作手册
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

sample_submit = pd.read_csv('sample_submit.csv')


# 训练数据及测试数据准备
all_cols = [f for f in train.columns if f not in ['customer_id','loan_default']]

x_train = train[all_cols]
x_test = test[all_cols]

y_train = train['loan_default']


# 作为baseline部分仅使用经典的**LightGBM**作为训练模型，我们还能尝试**XGBoost、CatBoost和NN（神经网络）**
def cv_model(clf, train_x, train_y, test_x, clf_name='lgb'):
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'min_child_weight': 5,
            'num_leaves': 2 ** 7,
            'lambda_l2': 10,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 4,
            'learning_rate': 0.01,
            'seed': 2021,
            'nthread': 28,
            'n_jobs':-1,
            'silent': True,
            'verbose': -1,
        }

        model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], verbose_eval=500,early_stopping_rounds=200)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)

        # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])

        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test



lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test)


# 预测结果
sample_submit['loan_default'] = lgb_test
sample_submit['loan_default'] = sample_submit['loan_default'].apply(lambda x:1 if x>0.25 else 0).values
sample_submit.to_csv('baseline_result.csv', index=False)




