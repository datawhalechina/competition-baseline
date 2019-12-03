# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb

import os, sys, time, codecs, glob
from tqdm import tqdm, tqdm_notebook

from sklearn.metrics import log_loss, classification_report
from sklearn.externals.joblib import Parallel, delayed

def timestamp_datetime(value):
    value = time.localtime(value)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', value)
    return dt

def feature_agg(i, path):
    if i % 10000 == 0:
        print(i, path)
    # print(path)
    
    df = pd.read_hdf(path)
    df.reset_index(drop=True, inplace=True)
    df.sort_values(by='behavior_timestamp', inplace=True)
    df['behavior_timestamp'] = pd.to_datetime(df['behavior_timestamp'].apply(lambda x: timestamp_datetime(x / 1000)))
    df['video_uptime'] = pd.to_datetime(df['video_uptime'].apply(lambda x: timestamp_datetime(x)))
    
    featdict = {}
    featdict['user_id'] = df['user_id'].iloc[0]
    
    # user_male 用户性别
    featdict['user_male_male'] = 0
    featdict['user_male_female'] = 0
    featdict['user_male_nan'] = 0
    if df['user_male'].value_counts().index[0] == '男':
        featdict['user_male_male'] = 1
    elif df['user_male'].value_counts().index[0] == '女':
        featdict['user_male_female'] = 1
    else:
        featdict['user_male_nan'] = 1
    featdict['user_male_NUNIQUE'] = df['user_male'].nunique()
    featdict['user_male_COUNT'] = df['user_male'].count()
    featdict['user_male_NAN'] = sum(df['user_male'] == '-') / df.shape[0]
    
    # user_age 用户年龄
    age_dict = {
        '-': -1,
        '18以下': 16,
        '18-24': 20,
        '25-34': 27,
        '35-44': 40,
        '45-54': 50,
        '55-64': 60,
        '65以上': 70,
    }
    featdict['user_age'] = age_dict[df['user_age'].value_counts().index[0]]
    if featdict['user_age'] == -1:
        featdict['user_age_nan'] = 1
    else:
        featdict['user_age_nan'] = 0
    featdict['user_age_NUNIQUE'] = df['user_age'].nunique()
    
    # user_edu 用户教育程度
    edu_dict = {
        '-': -1,
        '高中及以下': 1,
        '大专': 2,
        '本科及以上': 3,
    }
    featdict['user_edu'] = edu_dict[df['user_edu'].value_counts().index[0]]
    if featdict['user_edu'] == -1:
        featdict['user_edu_nan'] = 1
    else:
        featdict['user_edu_nan'] = 0
    featdict['user_edu_NUNIQUE'] = df['user_edu'].nunique()
    featdict['user_edu_NAN'] = sum(df['user_edu'] == '-') / df.shape[0]
    
    # user_install 用户安装渠道
    install_lbl = ['ctn_1', 'ctn_1005', 'ctn_1018', 'ctn_1029', 'ctn_1042', 'ctn_1043', 
     'ctn_112', 'ctn_13', 'ctn_14', 'ctn_144', 'ctn_149', 'ctn_15', 'ctn_150',
     'ctn_151', 'ctn_159', 'ctn_16', 'ctn_160', 'ctn_161', 'ctn_163', 'ctn_17',
     'ctn_185', 'ctn_188', 'ctn_2', 'ctn_20', 'ctn_202', 'ctn_23', 'ctn_239',
     'ctn_24', 'ctn_240', 'ctn_27', 'ctn_29', 'ctn_308', 'ctn_341', 'ctn_358',
     'ctn_368', 'ctn_371', 'ctn_430', 'ctn_484', 'ctn_487', 'ctn_5', 'ctn_55',
     'ctn_664', 'ctn_666', 'ctn_745', 'ctn_746', 'ctn_772', 'ctn_875', 'ctn_89',
     'ctn_921', 'ctn_110']
    ctns = df['user_install'].unique()
    for ctn in install_lbl:
        if ctn in ctns:
            featdict[ctn] = 1
        else:
            featdict[ctn] = 0
    featdict['user_install_NUNIQUE'] = df['user_install'].nunique()
    
    # video_id 
    featdict['video_id_NUNIQUE'] = df['video_id'].nunique()
    featdict['video_class_NUNIQUE'] = df['video_class'].nunique()
    featdict['video_duration_NUNIQUE'] = df['video_duration'].nunique()
    
    # behavior_show 展现的比例
    # behavior_show 连续出现的比例
    show_counts = df['behavior_show'].value_counts()
    featdict['behavior_show_flag'] = int('1' in show_counts.index)
    featdict['behavior_show_keep'] = sum(pd.Series(df[df['behavior_show'] == '1'].index).diff(1) == 1)
    if featdict['behavior_show_flag']:
        featdict['behavior_show_ratio'] = show_counts['1'] / show_counts.sum()
        featdict['behavior_show_keep_ratio'] = featdict['behavior_show_keep'] / show_counts['1']
    else:
        featdict['behavior_show_ratio'] = 0
        featdict['behavior_show_keep_ratio'] = 0
    
    # 前10/20/50 后10/20/50 behavior_show 展现的比例
    if df.shape[0] < 10:
        featdict['behavior_show_first10_ratio'] = 0
        featdict['behavior_show_last10_ratio'] = 0
    else:
        featdict['behavior_show_first10_ratio'] = int('1' in df.iloc[:10]['behavior_show'].values)
        featdict['behavior_show_last10_ratio'] = int('1' in df.iloc[-10:]['behavior_show'].values)

    if df.shape[0] < 20:
        featdict['behavior_show_first20_ratio'] = 0
        featdict['behavior_show_last20_ratio'] = 0
    else:
        featdict['behavior_show_first20_ratio'] = int('1' in df.iloc[:20]['behavior_show'].values)
        featdict['behavior_show_last20_ratio'] = int('1' in df.iloc[-20:]['behavior_show'].values)

    if df.shape[0] < 50:
        featdict['behavior_show_first50_ratio'] = 0
        featdict['behavior_show_last50_ratio'] = 0
    else:
        featdict['behavior_show_first50_ratio'] = int('1' in df.iloc[:50]['behavior_show'].values)
        featdict['behavior_show_last50_ratio'] = int('1' in df.iloc[-50:]['behavior_show'].values)    
    
    # behavior_click 点击的比例
    # behavior_click 连续的比例
    click_counts = df['behavior_click'].value_counts()
    featdict['behavior_click_flag'] = int('1' in click_counts.index)
    featdict['behavior_click_keep'] = sum(pd.Series(df[df['behavior_click'] == '1'].index).diff(1) == 1)
    if featdict['behavior_click_flag']:
        featdict['behavior_click_ratio'] = click_counts['1'] / click_counts.sum()
        if '0' in click_counts.index:
            featdict['behavior_show_notclick_ratio'] = click_counts['1'] / (click_counts['1'] + click_counts['0'])
            featdict['behavior_click_keep_ratio'] = featdict['behavior_click_keep']/ click_counts['1']
        else:
            featdict['behavior_show_notclick_ratio'] = 0
            featdict['behavior_click_keep_ratio'] = 0
    else:
        featdict['behavior_click_ratio'] = 0
        featdict['behavior_show_notclick_ratio'] = 0
    
    featdict['behavior_recommend_NUNIQUE'] = df['behavior_recommend'].nunique()
    
    df_tmp = df[df['behavior_playback'] != '-']
    if df_tmp.shape[0] == 0:
        featdict['behavior_playback_mean'] = 0
        featdict['behavior_playback_mean2'] = 0
        featdict['behavior_playback_max'] = 0
        featdict['behavior_playback_sum'] = 0
        featdict['behavior_playback_ratio'] = 0
        
        featdict['behavior_comment_ratio'] = 0
        featdict['behavior_like_ratio'] = 0
        featdict['behavior_forard_ratio'] = 0
        
        featdict['behavior_playback_video_mean'] = 0
        featdict['behavior_playback_video_min'] = 0
        featdict['behavior_playback_video_max'] = 0
    else:
        featdict['behavior_playback_mean'] = df_tmp['behavior_playback'].astype(float).mean()
        featdict['behavior_playback_mean2'] = df_tmp[df_tmp['behavior_playback'] != 0]['behavior_playback'].astype(float).mean()
        featdict['behavior_playback_max'] = df_tmp['behavior_playback'].astype(float).max()
        featdict['behavior_playback_sum'] = df_tmp['behavior_playback'].astype(float).sum()
        featdict['behavior_playback_ratio'] = df_tmp[df_tmp['behavior_playback'] != 0].shape[0] / df_tmp.shape[0]
        
        featdict['behavior_comment_ratio'] = df_tmp[df_tmp['behavior_comment'] == 1].shape[0] / df_tmp.shape[0]   
        featdict['behavior_like_ratio'] = df_tmp[df_tmp['behavior_like'] == 1].shape[0] / df_tmp.shape[0]
        featdict['behavior_forard_ratio'] = df_tmp[df_tmp['behavior_forard'] == 1].shape[0] / df_tmp.shape[0]
        
        df_tmp['behavior_playback_div_video_duration'] = df_tmp['behavior_playback'].astype(float) / df_tmp['video_duration']
        featdict['behavior_playback_video_mean'] = df_tmp['behavior_playback_div_video_duration'].mean()
        featdict['behavior_playback_video_min'] = df_tmp['behavior_playback_div_video_duration'].max()
        featdict['behavior_playback_video_max'] = df_tmp['behavior_playback_div_video_duration'].min()
    
    featdict['behavior_playback_sum_minute'] = featdict['behavior_playback_sum'] % 60
    featdict['behavior_playback_sum_hour'] = featdict['behavior_playback_sum'] % 3600
    
    featdict['behavior_timestamp_month_NUNIQUE'] = df['behavior_timestamp'].dt.month.nunique()
    featdict['behavior_timestamp_day_NUNIQUE'] = df['behavior_timestamp'].dt.day.nunique()
    featdict['behavior_timestamp_hour_NUNIQUE'] = df['behavior_timestamp'].dt.hour.nunique()
    featdict['behavior_timestamp_minute_NUNIQUE'] = df['behavior_timestamp'].dt.minute.nunique()
    
    featdict['behavior_playback_mean_day'] = featdict['behavior_playback_sum'] / featdict['behavior_timestamp_day_NUNIQUE']
    featdict['behavior_playback_mean_hour'] = featdict['behavior_playback_sum'] / featdict['behavior_timestamp_hour_NUNIQUE']
    
    hour_unique = df['behavior_timestamp'].dt.hour.unique() 
    for hour in range(24):
        if hour in hour_unique:
            featdict['behavior_timestamp_hour' + str(hour)] = 1
        else:
            featdict['behavior_timestamp_hour' + str(hour)] = 0        
    
    featdict['behavior_timestamp_likeday_NUNIQUE'] = df['behavior_timestamp'].dt.minute.nunique()
    
    # 使用时间
    behavior_timestamp = df['behavior_timestamp'].iloc[-1] - df['behavior_timestamp'].iloc[0]
    featdict['behavior_timestamp_second'] = behavior_timestamp.seconds
    featdict['behavior_timestamp_minute'] = behavior_timestamp.seconds % 60
    featdict['behavior_timestamp_hour'] = behavior_timestamp.seconds % 3600
    featdict['behavior_timestamp_day'] = behavior_timestamp.seconds % 86400
    if featdict['behavior_timestamp_second'] == 0:
        featdict['behavior_playback_time_ratio'] = 0
    else:
        featdict['behavior_playback_time_ratio'] = featdict['behavior_playback_sum'] / featdict['behavior_timestamp_second']
    
    # df_tmp = df[df['behavior_click'] != '-']
    # featdict['behavior_timestamp_click_month_NUNIQUE'] = df_tmp['behavior_timestamp'].dt.month.nunique()
    # featdict['behavior_timestamp_click_day_NUNIQUE'] = df_tmp['behavior_timestamp'].dt.day.nunique()
    # featdict['behavior_timestamp_click_hour_NUNIQUE'] = df_tmp['behavior_timestamp'].dt.hour.nunique()
    # featdict['behavior_timestamp_click_minute_NUNIQUE'] = df_tmp['behavior_timestamp'].dt.minute.nunique()
    
    return featdict

train_id = pd.read_csv('./train_id.csv')
test_id = pd.read_csv('./test_id.csv')

train_id = pd.read_csv('./train_id.csv')
test_id = pd.read_csv('./test_id.csv')

train_feat = Parallel(n_jobs=30)(delayed(feature_agg)(i, './train/'+id+'.hdf') for i, id in enumerate(train_id['user_id'].iloc[:]))
test_feat = Parallel(n_jobs=30)(delayed(feature_agg)(i, './test/'+id+'.hdf') for i, id in enumerate(test_id['user_id'].iloc[:]))
train_feat = pd.DataFrame(train_feat)
test_feat = pd.DataFrame(test_feat)

train_feat = pd.merge(train_feat, train_id, on='user_id', how='left')

params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'max_depth': -1,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'binary',
    'n_estimators': 2000,
    'metric': 'auc',
    # 'num_class': 6,
    'feature_fraction': .85,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
}

# cv_results1 = lgb.cv(
#         params,
#         lgb.Dataset(train_feat.drop(['user_id', 'label'], axis=1).values, label=train_feat['label'].values),
#         num_boost_round=200,
#         nfold=7, verbose_eval=False,
#         early_stopping_rounds=200,
# )
# print('CV AUC: ', len(cv_results1['auc-mean']), cv_results1['auc-mean'][-1])

# clf = lgb.train(
#         params,
#         lgb.Dataset(train_feat.drop(['user_id', 'label'], axis=1).values, label=train_feat['label'].values),
#         num_boost_round=1000)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, shuffle = True)
eval_fun = roc_auc_score

def run_oof(clf, X_train, y_train, X_test, kf):
    print(clf)
    preds_train = np.zeros((len(X_train)), dtype = np.float)
    preds_test = np.zeros((len(X_test)), dtype = np.float)
    train_loss = []; test_loss = []

    i = 1
    for train_index, test_index in kf.split(X_train, y_train):
        x_tr = X_train[train_index]; x_te = X_train[test_index]
        y_tr = y_train[train_index]; y_te = y_train[test_index]
        clf.fit(x_tr, y_tr, eval_set = [(x_te, y_te)], early_stopping_rounds = 500, verbose = False)
        
        train_loss.append(eval_fun(y_tr, clf.predict_proba(x_tr)[:, 1]))
        test_loss.append(eval_fun(y_te, clf.predict_proba(x_te)[:, 1]))

        preds_train[test_index] = clf.predict_proba(x_te)[:, 1]
        preds_test += clf.predict_proba(X_test)[:, 1]

        print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], test_loss[-1], np.mean(test_loss)))
        print('-' * 50)
        i += 1
    print('Train: ', train_loss)
    print('Val: ', test_loss)
    print('-' * 50)
    print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(np.mean(train_loss), np.mean(test_loss)))
    preds_test /= n_fold
    return preds_train, preds_test

params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'max_depth': -1,
    'lambda_l1': 5,
    'boosting': 'gbdt',
    'objective': 'binary',
    'n_estimators': 5000,
    'metric': 'auc',
    # 'num_class': 6,
    'feature_fraction': .75,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
}

train_pred, test_pred = run_oof(lgb.LGBMClassifier(**params), 
                                train_feat.drop(['user_id', 'label'], axis=1).values, 
                                train_feat['label'].values, 
                                test_feat.drop(['user_id'], axis=1).values, 
                                skf)

test_feat['label'] = test_pred
test_feat[['user_id', 'label']].to_csv('baseline.csv', index=None, header=None)