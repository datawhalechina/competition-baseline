import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submit = pd.read_csv('sample_submit.csv')

df = pd.concat([train, test], axis=0, ignore_index=True)

def lag_feature_adv(df, lags, col):
    '''
    历史N周平移特征
    '''
    tmp = df[['week','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['week','shop_id','item_id', col+'_lag_'+str(i)+'_adv']
        shifted['week'] += i
        df = pd.merge(df, shifted, on=['week','shop_id','item_id'], how='left')
        df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv']
    return df

df = lag_feature_adv(df, [1, 2, 3], 'weekly_sales')

x_train = df[df.week < 33].drop(['weekly_sales'], axis=1)
y_train = df[df.week < 33]['weekly_sales']
x_test = df[df.week == 33].drop(['weekly_sales'], axis=1)


def cv_model(clf, train_x, train_y, test_x, clf_name='lgb'):
    folds = 5
    seed = 1024
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])
     
    categorical_feature = ['shop_id','item_id','item_category_id']
    cv_scores = []

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'mse',
            'metric': 'mse',
            'min_child_weight': 5,
            'num_leaves': 2 ** 7,
            'lambda_l2': 10,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 4,
            'learning_rate': 0.05,
            'seed': 1024,
            'n_jobs':-1,
            'silent': True,
            'verbose': -1,
        }

        model = clf.train(params, train_matrix, 5000, valid_sets=[train_matrix, valid_matrix], 
                          categorical_feature = categorical_feature,
                          verbose_eval=500,early_stopping_rounds=200)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)

        train[valid_index] = val_pred
        test += test_pred / kf.n_splits
        cv_scores.append(mean_squared_error(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test)


sample_submit['weekly_sales'] = lgb_test
sample_submit['weekly_sales'] = sample_submit['weekly_sales'].apply(lambda x:x if x>0 else 0).values
sample_submit.to_csv('baseline_result.csv', index=False)
