# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import lightgbm as lgb

import os, sys, time, codecs, glob
from tqdm import tqdm, tqdm_notebook

from sklearn.metrics import log_loss, classification_report
from sklearn.externals.joblib import Parallel, delayed
from collections import Counter

def timestamp_datetime(value):
    value = time.localtime(value)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', value)
    return dt

installs = pd.read_csv('./data/install_counts.csv')
install_count = np.zeros(installs.shape[0])

install_count[0:5] = 6
install_count[5:10] = 5
install_count[10:30] = 4
install_count[30:80] = 3
install_count[80:250] = 2
install_count[250:500] = 1
install_count[500:] = 0

install_mean = pd.read_csv('./data/install_mean.csv')

# 求解得到列表连续的上升元素的区间
# 返回 [长度，开始，结束]
def longUpperStep(arr):
    arr = list(arr)
    
    lenx, idx = [], []
    idx1 = 0
    while idx1 < len(arr):
        end = idx1 + 1
        for idx2 in range(idx1, len(arr)):
            if idx1 == idx2:
                continue

            if arr[idx1:idx2] == range(arr[idx1], arr[idx2]):
                end = idx2
                continue

        if end != idx1 + 1:
            lenx.append(end - idx1)
            idx.append([idx1, end])
            idx1 = end
        elif end < len(arr) and arr[idx1] + 1 == arr[end]:
            lenx.append(end - idx1)
            idx.append([idx1, end])
            idx1 = end
        else:
            idx1 += 1
    return lenx, idx

def feature_agg(i, path):
    if i % 10000 == 0:
        print(i, path)

    df = pd.read_csv(path,  dtype={'user_id':object, 'user_male':object, 'user_age':object,
                            'user_edu':object, 'user_district':object, 'user_install':object,
                            'video_id':object, 'video_class':object, 'video_tag':object,
                            'video_creator':object, 'video_uptime':int, 'video_duration':int,
                            'behavior_show':object, 'behavior_click':object, 'behavior_recommend':object,
                            'behavior_playback':object, 'behavior_timestamp':int, 'behavior_comment':object,
                            'behavior_like':object, 'behavior_forard':object})

    df.sort_values(by='behavior_timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['behavior_timestamp'] = pd.to_datetime(df['behavior_timestamp'].apply(lambda x: timestamp_datetime(x / 1000)))
    df['video_uptime'] = pd.to_datetime(df['video_uptime'].apply(lambda x: timestamp_datetime(x)))

    featdict = {}
    featdict['user_id'] = df['user_id'].iloc[0]

    ############################################################################
    # 用户基本信息
    ############################################################################
    # user 用户记录的条数
    featdict['user_count'] = df['user_male'].count()

    # user_male 用户性别编码，用出现次数最多的编码
    featdict['user_male_male'] = 0
    featdict['user_male_female'] = 0
    featdict['user_male_nan'] = 0
    if df['user_male'].value_counts().index[0] == '男':
        featdict['user_male_male'] = 1
    elif df['user_male'].value_counts().index[0] == '女':
        featdict['user_male_female'] = 1
    else:
        featdict['user_male_nan'] = 1

    # user_male 用户性别种类个数
    featdict['user_male_NUNIQUE'] = df['user_male'].nunique()
    # user_male 用户性别缺失比例
    featdict['user_male_NAN'] = sum(df['user_male'] == '-') / df.shape[0]

    # user_age 用户年龄编码
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
    # 使用出现次数最多的最为年龄编码
    featdict['user_age'] = age_dict[df['user_age'].value_counts().index[0]]
    
    age_dict = {
        '-': 0,
        '18以下': 6,
        '18-24': 2,
        '25-34': 3,
        '35-44': 4,
        '45-54': 7,
        '55-64': 5,
        '65以上': 1,
    }
    # 年龄 CTR
    featdict['user_age_CTR'] = age_dict[df['user_age'].value_counts().index[0]]
    
    # 年龄缺失编码
    if featdict['user_age'] == -1:
        featdict['user_age_nan'] = 1
    else:
        featdict['user_age_nan'] = 0
    
    featdict['user_age_NAN'] = sum(df['user_age'] == '-') / df.shape[0]
    featdict['user_age_NUNIQUE'] = df['user_age'].nunique()
    
    df['user_age'] = df['user_age'].apply(lambda x: age_dict[x])
    if featdict['user_age_NAN'] == 1:
        featdict['user_age_MIN'] = 0
        featdict['user_age_MAX'] = 0
        featdict['user_age_MEAN'] = 0
        featdict['user_age_STD'] = 0
        # featdict['user_age_PTP'] = 0
    else:
        featdict['user_age_MIN'] = df[df['user_age'] != -1]['user_age'].mean()
        featdict['user_age_MAX'] = df[df['user_age'] != -1]['user_age'].max()
        featdict['user_age_MEAN'] = df[df['user_age'] != -1]['user_age'].mean()
        featdict['user_age_STD'] = df[df['user_age'] != -1]['user_age'].std()
        # featdict['user_age_PTP'] = df[df['user_age'] != -1]['user_age'].ptp()
    
    # user_age 用户年龄编码 留存率分级
    
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
    
    # user_edu 用户教育程度 留存率分级
    
    featdict['user_age*edu'] = featdict['user_edu'] * featdict['user_age']
    
    featdict['user_edu_NAN'] = sum(df['user_edu'] == '-') / df.shape[0]
    df['user_edu'] = df['user_edu'].apply(lambda x: edu_dict[x])
    featdict['user_edu_NUNIQUE'] = df['user_edu'].nunique()
    featdict['user_edu_MAX'] = df['user_edu'].max()
    
    # user_install 用户安装渠道
    install_lbl = ['ctn_1005', 'ctn_1018', 'ctn_1029', 'ctn_1042', 'ctn_1043',
     'ctn_112', 'ctn_13', 'ctn_14', 'ctn_144', 'ctn_149', 'ctn_15', 'ctn_150',
     'ctn_151', 'ctn_159', 'ctn_16', 'ctn_160', 'ctn_161', 'ctn_163', 'ctn_17',
     'ctn_185', 'ctn_188', 'ctn_2', 'ctn_20', 'ctn_202', 'ctn_23', 'ctn_239',
     'ctn_24', 'ctn_240', 'ctn_27', 'ctn_29', 'ctn_308', 'ctn_341', 'ctn_358',
     'ctn_368', 'ctn_371', 'ctn_430', 'ctn_484', 'ctn_487', 'ctn_5', 'ctn_55',
     'ctn_664', 'ctn_666', 'ctn_745', 'ctn_746', 'ctn_772', 'ctn_875', 'ctn_89',
     'ctn_921', 'ctn_110',
    'ctn_1005', 'ctn_542',
       'ctn_144', 'ctn_182', 'ctn_371', 'ctn_1042', 'ctn_1043', 'ctn_484',
       'ctn_358', 'ctn_880', 'ctn_1029', 'ctn_792', 'ctn_543', 'ctn_790',
       'ctn_28', 'ctn_489', 'ctn_469', 'ctn_77', 'ctn_113', 'ctn_960',
       'ctn_480', 'ctn_308', 'ctn_572', 'ctn_44', 'ctn_982', 'ctn_716',
       'ctn_186', 'ctn_154', 'ctn_474', 'ctn_436', 'ctn_520', 'ctn_187',
       'ctn_338', 'ctn_309', 'ctn_15', 'ctn_984', 'ctn_134', 'ctn_76',
    'ctn_704', 'ctn_415', 'ctn_830', 'ctn_710', 'ctn_391', 'ctn_823',
       'ctn_971', 'ctn_22', 'ctn_503', 'ctn_21', 'ctn_918', 'ctn_1047',
       'ctn_516', 'ctn_423', 'ctn_972', 'ctn_344', 'ctn_79', 'ctn_397',
       'ctn_457', 'ctn_765', 'ctn_665', 'ctn_327', 'ctn_158', 'ctn_701',
       'ctn_671', 'ctn_315', 'ctn_25', 'ctn_877', 'ctn_509', 'ctn_719',
       'ctn_350', 'ctn_718', 'ctn_831', 'ctn_471', 'ctn_789', 'ctn_162',
       'ctn_700', 'ctn_224', 'ctn_900', 'ctn_871', 'ctn_726', 'ctn_170',
       'ctn_1037', 'ctn_316', 'ctn_452', 'ctn_715', 'ctn_31', 'ctn_87',
       'ctn_1036', 'ctn_513', 'ctn_121', 'ctn_794', 'ctn_791', 'ctn_156',
       'ctn_938', 'ctn_782', 'ctn_366', 'ctn_349', 'ctn_6', 'ctn_179',
       'ctn_372', 'ctn_1017', 'ctn_216', 'ctn_486', 'ctn_928', 'ctn_959',
    ]
    
    ctns = df['user_install'].unique()
    df_tmp = df[['user_install', 'behavior_recommend']].drop_duplicates()
    arr = (df_tmp['user_install'] + df_tmp['behavior_recommend']).values
    
    for ctn in install_lbl:
        if ctn in ctns:
            featdict['user_install_' + ctn] = 1
            
            if 'ctn_1' in ctns:
                featdict['user_install2_' + ctn] = 1
                
                if 'ctn_1rec_type_1' in arr:
                    featdict['ctn_1rec_type_1'] = 1
                else:
                    featdict['ctn_1rec_type_1'] = 0
                    
                if 'ctn_1rec_type_2' in arr:
                    featdict['ctn_1rec_type_2'] = 1
                else:
                    featdict['ctn_1rec_type_2'] = 0
                    
                if 'ctn_1rec_type_3' in arr:
                    featdict['ctn_1rec_type_3'] = 1
                else:
                    featdict['ctn_1rec_type_3'] = 0
                
            else:
                featdict['user_install2_' + ctn] = 0
                
                featdict['ctn_1rec_type_1'] = 0
                featdict['ctn_1rec_type_2'] = 0
                featdict['ctn_1rec_type_3'] = 0
                
        else:
            featdict['user_install_' + ctn] = 0
            featdict['user_install2_' + ctn] = 1
            
            featdict['ctn_1rec_type_1'] = 0
            featdict['ctn_1rec_type_2'] = 0
            featdict['ctn_1rec_type_3'] = 0

    featdict['user_install_NUNIQUE'] = df['user_install'].nunique()

    # user_install 用户安装渠道 留存率分级
    if df['user_install'].iloc[0] in install_mean['user_install'].values:
        featdict['install_mean'] = np.where(df['user_install'].iloc[0] == install_mean['user_install'])[0][0]
    else:
        featdict['install_mean'] = 1000
    
    install_set = []
    for install in df['user_install'].unique():
        if install in install_mean['user_install'].values:
            install_set.append(np.where(install == install_mean['user_install'])[0][0])
        else:
            install_set.append(1000)
    featdict['install_mean_mean'] = np.mean(install_set)
    featdict['install_mean_min'] = np.min(install_set)
    featdict['install_mean_max'] = np.max(install_set)
    
    # user_install 用户安装渠道 COUNT分级
    if df['user_install'].iloc[0] in installs['user_install'].values:
        featdict['install_first_count'] = install_count[np.where(df['user_install'].iloc[0] == installs['user_install'])[0][0]]
    else:
        featdict['install_first_count'] = 0
    
    if df['user_install'].iloc[-1] in installs['user_install'].values:
        featdict['install_last_count'] = install_count[np.where(df['user_install'].iloc[-1] == installs['user_install'])[0][0]]
    else:
        featdict['install_last_count'] = 0
    
    if df['user_install'].value_counts().index[0] in installs['user_install'].values:
        featdict['install_count_MOSTVALUE'] = install_count[np.where(df['user_install'].value_counts().index[0] == installs['user_install'])[0][0]]
    else:
        featdict['install_count_MOSTVALUE'] = 0
    
    ############################################################################
    # 用户（基本信息）与视频的交叉特征
    ############################################################################
    # 用户视频个数
    # featdict['video_NUNIQUE'] = df['video_id'].nunique()
    # 用户视频类别格式
    # featdict['video_class_NUNIQUE'] = df['video_class'].nunique()
    
    # featdict['video_show_NUNIQUE'] = df[df['behavior_show'] == '1']['video_id'].nunique()
    featdict['video_show_class_NUNIQUE'] = df[df['behavior_show'] == '1']['video_class'].nunique()

    featdict['video_click_NUNIQUE'] = df[df['behavior_click'] == '1']['video_id'].nunique()
    featdict['video_click_class_NUNIQUE'] = df[df['behavior_click'] == '1']['video_class'].nunique()
    
    featdict['video_click_insall_NUNIQUE'] = df[df['behavior_click'] == '1']['user_install'].nunique()
    
    # featdict['video_duration_NUNIQUE'] = df['video_duration'].nunique()

    video_class_dict = ['category_149', 'category_152', 'category_169', 'category_178',
       'category_197', 'category_103', 'category_136', 'category_75',
       'category_109', 'category_158']

    for c in video_class_dict:
        if c in df[df['behavior_show'] == '1']['video_class'].values:
            featdict[c + '_show'] = 1
        else:
            featdict[c + '_show'] = 0
    
    # 用户是否观看视频多次
    # featdict['user_videio_>2'] = int(df['video_id'].value_counts().max() >= 2)
    # featdict['user_videio_show_>2'] = int(df[df['behavior_show'] == '1']['video_id'].value_counts().max() >= 2)
    # featdict['user_videio_>2'] = int(df[df['behavior_click'] == '1']['video_id'].value_counts().max() >= 2)
    
    # 用户是否观看同一个作者多次
    # featdict['user_videio_creator_>2'] = int(df['video_creator'].value_counts().max() >= 2)
    # featdict['user_videio_creator_show_>2'] = int(df[df['behavior_show'] == '1']['video_creator'].value_counts().max() >= 2)
    # featdict['user_videio_creator_>2'] = int(df[df['behavior_click'] == '1']['video_creator'].value_counts().max() >= 2)
    
    # 是否观看同种tag视频多次
    tags = '$'.join(df[df['behavior_show'] == '1']['video_tag']).split('$')
    if len(tags) > 2:
        tags = Counter([x for x in tags if x != ''])
        featdict['user_videio_tags'] = int(tags.most_common(1)[0][1] > 1)
    else:
        featdict['user_videio_tags'] = 0
    
    featdict['user_video_same_week'] = sum(df['video_uptime'].dt.week == df['behavior_timestamp'].dt.week) / df.shape[0]
    featdict['user_video_same_month'] = sum(df['video_uptime'].dt.month == df['behavior_timestamp'].dt.month) / df.shape[0]
    
    featdict['user_video_month_nunique'] = df['video_uptime'].dt.month.nunique()
    # featdict['user_video_day_nunique'] = df['video_uptime'].dt.day.nunique()
    
    featdict['user_video_month_2_behavior_min'] = (df['behavior_timestamp'].dt.month - df['video_uptime'].dt.month).min()
    featdict['user_video_month_2_behavior_max'] = (df['behavior_timestamp'].dt.month - df['video_uptime'].dt.month).max()
    # featdict['user_video_month_2_behavior_mean'] = (df['behavior_timestamp'].dt.month - df['video_uptime'].dt.month).mean()
    
    featdict['user_video_day_2_behavior_min'] = (df['behavior_timestamp'].dt.dayofyear - df['video_uptime'].dt.dayofyear).min()
    featdict['user_video_day_2_behavior_max'] = (df['behavior_timestamp'].dt.dayofyear - df['video_uptime'].dt.dayofyear).max()
    # featdict['user_video_day_2_behavior_mean'] = (df['behavior_timestamp'].dt.dayofyear - df['video_uptime'].dt.dayofyear).mean()
    
    df_tmp = df[df['behavior_show'] == '1']
    featdict['user_video_show_same_week'] = sum(df_tmp['video_uptime'].dt.week == df_tmp['behavior_timestamp'].dt.week) / df.shape[0]
    
    # df_tmp = df[df['behavior_click'] == '1']
    # featdict['user_video_click_same_week'] = sum(df_tmp['video_uptime'].dt.week == df_tmp['behavior_timestamp'].dt.week) / df.shape[0]
    
    ############################################################################
    # 用户（基本信息）与行为特征
    ############################################################################

    # behavior_show 展现的比例
    # behavior_show 连续出现的比例
    show_counts = df['behavior_show'].value_counts()
    featdict['behavior_show_flag'] = int('1' in show_counts.index)
    featdict['behavior_show_keep'] = sum(pd.Series(df[df['behavior_show'] == '1'].index).diff(1) == 1)
    if featdict['behavior_show_flag']:
        # featdict['behavior_show_ratio'] = show_counts['1'] / show_counts.sum()
        featdict['behavior_show_keep_ratio'] = featdict['behavior_show_keep'] / show_counts['1']
        # featdict['behavior_show_last_index'] = df[df['behavior_show'] == '1'].index[-1] / df.shape[0]
    else:
        # featdict['behavior_show_ratio'] = 0
        featdict['behavior_show_keep_ratio'] = 0
        # featdict['behavior_show_last_index'] = 0
    
    
    # 前10/20/50 后10/20/50 behavior_show 展现的比例
    
    featdict['behavior_show_last1_ratio'] = int('1' in df.iloc[-1:]['behavior_show'].values)
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
        # featdict['behavior_click_ratio'] = click_counts['1'] / click_counts.sum()
        if '0' in click_counts.index:
            # featdict['behavior_show_notclick_ratio'] = click_counts['1'] / (click_counts['1'] + click_counts['0'])
            featdict['behavior_click_keep_ratio'] = featdict['behavior_click_keep']/ click_counts['1']
        else:
            # featdict['behavior_show_notclick_ratio'] = 0
            featdict['behavior_click_keep_ratio'] = 0
        # featdict['behavior_click_last_index'] = df[df['behavior_click'] == '1'].index[-1] / df.shape[0]
    else:
        featdict['behavior_click_keep_ratio'] = 0
        # featdict['behavior_click_last_index'] = 0
        # featdict['behavior_click_ratio'] = 0
        # featdict['behavior_show_notclick_ratio'] = 0

    featdict['behavior_click_last1_ratio'] = int('1' in df.iloc[-1:]['behavior_click'].values)
    if df.shape[0] < 10:
        featdict['behavior_click_first10_ratio'] = 0
        # featdict['behavior_click_last10_ratio'] = 0
    else:
        featdict['behavior_click_first10_ratio'] = int('1' in df.iloc[:10]['behavior_click'].values)
        # featdict['behavior_click_last10_ratio'] = int('1' in df.iloc[-10:]['behavior_click'].values)
        
    # 不同 behavior_recommend 情况下的统计
    featdict['behavior_recommend_NUNIQUE'] = df['behavior_recommend'].nunique()
    
    df_tmp = df['behavior_recommend'].value_counts()
    if 'rec_type_1' in df_tmp.index:
        featdict['rec_type_1_radio'] = df_tmp['rec_type_1'] / df.shape[0]
        # featdict['rec_type_1_show_radio'] = df[(df['behavior_show'] == '1') & (df['behavior_recommend'] == 'rec_type_1')].shape[0] / (df[df['behavior_show'] == '1'].shape[0]+1)
    else:
        featdict['rec_type_1_radio'] = 0
        # featdict['rec_type_1_show_radio'] = 0
    if 'rec_type_2' in df_tmp.index:
        featdict['rec_type_2_radio'] = df_tmp['rec_type_2'] / df.shape[0]
        # featdict['rec_type_2_show_radio'] = df[(df['behavior_show'] == '1') & (df['behavior_recommend'] == 'rec_type_2')].shape[0] / (df[df['behavior_show'] == '1'].shape[0]+1)
    else:
        featdict['rec_type_2_radio'] = 0
        # featdict['rec_type_2_show_radio'] = 0
    if 'rec_type_3' in df_tmp.index:
        featdict['rec_type_3_radio'] = df_tmp['rec_type_3'] / df.shape[0]
        # featdict['rec_type_3_show_radio'] = df[(df['behavior_show'] == '1') & (df['behavior_recommend'] == 'rec_type_3')].shape[0] / (df[df['behavior_show'] == '1'].shape[0]+1)
    else:
        featdict['rec_type_3_radio'] = 0
        # featdict['rec_type_3_show_radio'] = 0
    
    
    df_tmp = df[df['behavior_playback'] != '-']
    if df_tmp.shape[0] == 0:
        featdict['behavior_playback_mean'] = 0
        featdict['behavior_playback_mean2'] = 0
        # featdict['behavior_playback_max'] = 0
        featdict['behavior_playback_sum'] = 0
        # featdict['behavior_playback_ratio'] = 0

        featdict['behavior_comment_ratio'] = 0
        featdict['behavior_like_ratio'] = 0
        # featdict['behavior_forard_ratio'] = 0

        featdict['behavior_playback_video_mean'] = 0
        # featdict['behavior_playback_video_max'] = 0
        featdict['behavior_playback_video_min'] = 0
        featdict['behavior_playback_video_>1'] = 0
    else:
        featdict['behavior_playback_mean'] = df_tmp['behavior_playback'].astype(float).mean()
        featdict['behavior_playback_mean2'] = df_tmp[df_tmp['behavior_playback'] != 0]['behavior_playback'].astype(float).mean()
        # featdict['behavior_playback_max'] = df_tmp['behavior_playback'].astype(float).max()
        featdict['behavior_playback_sum'] = df_tmp['behavior_playback'].astype(float).sum()
        # featdict['behavior_playback_ratio'] = df_tmp[df_tmp['behavior_playback'] != 0].shape[0] / df_tmp.shape[0]

        featdict['behavior_comment_ratio'] = df_tmp[df_tmp['behavior_comment'] == 1].shape[0] / df_tmp.shape[0]
        featdict['behavior_like_ratio'] = df_tmp[df_tmp['behavior_like'] == 1].shape[0] / df_tmp.shape[0]
        # featdict['behavior_forard_ratio'] = df_tmp[df_tmp['behavior_forard'] == 1].shape[0] / df_tmp.shape[0]

        df_tmp['behavior_playback_div_video_duration'] = df_tmp['behavior_playback'].astype(float) / (df_tmp['video_duration'] + 1.0)
        featdict['behavior_playback_video_mean'] = df_tmp['behavior_playback_div_video_duration'].mean()
        featdict['behavior_playback_video_max'] = df_tmp['behavior_playback_div_video_duration'].max()
        featdict['behavior_playback_video_min'] = df_tmp['behavior_playback_div_video_duration'].min()
        featdict['behavior_playback_video_>1'] = int(featdict['behavior_playback_video_max'])
        
        # featdict['behavior_playback_last_index'] = df_tmp.index[-1] / df.shape[0]
    
    df['behavior_playback'] = df['behavior_playback'].replace('-', '0.0').astype(float)
    df_tmp = df[df['behavior_playback'] > 0.0]
    if df_tmp.shape[0] == 0:
        featdict['behavior_playback2_mean'] = 0
        featdict['behavior_playback2_max'] = 0
        featdict['behavior_playback2_sum'] = 0
        
        featdict['behavior_comment2_ratio'] = 0
        featdict['behavior_like2_ratio'] = 0
        featdict['behavior_forard2_ratio'] = 0

        featdict['behavior_playback2_video_mean'] = 0
        featdict['behavior_playback2_video_max'] = 0
        featdict['behavior_playback2_video_min'] = 0
        featdict['behavior_playback2_video_>1'] = 0
    else:
        featdict['behavior_playback2_mean'] = df_tmp['behavior_playback'].astype(float).mean()
        featdict['behavior_playback2_max'] = df_tmp['behavior_playback'].astype(float).max()
        featdict['behavior_playback2_sum'] = df_tmp['behavior_playback'].astype(float).sum()
        
        featdict['behavior_comment2_ratio'] = df_tmp[df_tmp['behavior_comment'] == 1].shape[0] / df_tmp.shape[0]
        featdict['behavior_like2_ratio'] = df_tmp[df_tmp['behavior_like'] == 1].shape[0] / df_tmp.shape[0]
        featdict['behavior_forard2_ratio'] = df_tmp[df_tmp['behavior_forard'] == 1].shape[0] / df_tmp.shape[0]

        df_tmp['behavior_playback_div_video_duration'] = df_tmp['behavior_playback'].astype(float) / (df_tmp['video_duration'] + 1.0)
        featdict['behavior_playback2_video_mean'] = df_tmp['behavior_playback_div_video_duration'].mean()
        featdict['behavior_playback2_video_max'] = df_tmp['behavior_playback_div_video_duration'].max()
        featdict['behavior_playback2_video_min'] = df_tmp['behavior_playback_div_video_duration'].min()
        featdict['behavior_playback2_video_>1'] = int(featdict['behavior_playback2_video_max'])
    
    # 用户连续多少次点击但是没有行为
#     continuous_len, continuous_idx = longUpperStep(df[df['behavior_show'] == '1'].index)
#     if len(continuous_len) == 0:
#         featdict['behavior_show_continuous_noshow_max'] = 0
#         featdict['behavior_show_continuous_noshow_min'] = 0
#         featdict['behavior_show_continuous_noshow_mean'] = 0
#     else:
#         featdict['behavior_show_continuous_noshow_max'] = np.max(continuous_len)
#         featdict['behavior_show_continuous_noshow_min'] = np.min(continuous_len)
#         featdict['behavior_show_continuous_noshow_mean'] = np.mean(continuous_len)
    
#     continuous_len, continuous_idx = longUpperStep(df[df['behavior_click'] == '1'].index)
#     if len(continuous_len) == 0:
#         featdict['behavior_click_continuous_noshow_max'] = 0
#         featdict['behavior_click_continuous_noshow_min'] = 0
#         featdict['behavior_click_continuous_noshow_mean'] = 0
#     else:
#         featdict['behavior_click_continuous_noshow_max'] = np.max(continuous_len)
#         featdict['behavior_click_continuous_noshow_min'] = np.min(continuous_len)
#         featdict['behavior_click_continuous_noshow_mean'] = np.mean(continuous_len)
        
#     continuous_len, continuous_idx = longUpperStep(df_tmp[df_tmp['behavior_click'] > 0.0].index)
#     if len(continuous_len) == 0:
#         featdict['behavior_click_continuous_noshow_max'] = 0
#         featdict['behavior_click_continuous_noshow_min'] = 0
#         featdict['behavior_click_continuous_noshow_mean'] = 0
#     else:
#         featdict['behavior_click_continuous_noshow_max'] = np.max(continuous_len)
#         featdict['behavior_click_continuous_noshow_min'] = np.min(continuous_len)
#         featdict['behavior_click_continuous_noshow_mean'] = np.mean(continuous_len)
    
    featdict['behavior_show_last1_ratio'] = int(df['behavior_playback'].iloc[-1] > 0)
    if df.shape[0] < 10:
        featdict['behavior_playback_first10_ratio'] = 0
        featdict['behavior_playback_last10_ratio'] = 0
    else:
        featdict['behavior_playback_first10_ratio'] = sum(df['behavior_playback'].iloc[:10].replace('-', 0).astype(float) > 0) /10
        featdict['behavior_playback_last10_ratio'] = sum(df['behavior_playback'].iloc[-10:].replace('-', 0).astype(float) > 0) /10
    
    # 用户 behavior_click 对应的时间差
    df_tmp = df[df['behavior_click'] == '1']
    if df_tmp.shape[0] < 2:
        featdict['behavior_click_diff_mean'] = 0
        # featdict['behavior_click_diff_max'] = 0
        # featdict['behavior_click_diff_min'] = 0
    else:
        featdict['behavior_click_diff_mean'] = df_tmp['behavior_timestamp'].diff(1).mean().total_seconds()
        # featdict['behavior_click_diff_max'] = df_tmp['behavior_timestamp'].diff(1).max().total_seconds()
        # featdict['behavior_click_diff_min'] = df_tmp['behavior_timestamp'].diff(1).min().total_seconds()
    
    featdict['behavior_playback_sum_minute'] = featdict['behavior_playback_sum'] / 60
    featdict['behavior_playback_sum_hour'] = featdict['behavior_playback_sum'] / 3600
    
    featdict['behavior_timestamp_NUNIQUE'] = df['behavior_timestamp'].nunique()
    featdict['behavior_timestamp_NUNIQUE_divide_COUNT'] = featdict['behavior_timestamp_NUNIQUE'] / df.shape[0]
    
    # 有效运行时间的信息
    # arr = df['behavior_timestamp'].unique()
    # arr = (arr[1:] - arr[:-1]).astype('timedelta64[s]') / np.timedelta64(1, 's')
    
#     if len(arr[arr < 600]) == 0:
#         featdict['behavior_timestamp_DIFF1_SUM'] = 0
#         featdict['behavior_timestamp_DIFF1_MIN'] = 0
#         featdict['behavior_timestamp_DIFF1_MAX'] = 0
#         featdict['behavior_timestamp_DIFF1_MEAN'] = 0
#         featdict['behavior_timestamp_DIFF1_LEN'] = 0
#         # featdict['behavior_timestamp_DIFF1_show_ratio'] = 0
#     else:
#         arr1 = arr[arr < 600]
#         featdict['behavior_timestamp_DIFF1_SUM'] = np.sum(arr1)
#         featdict['behavior_timestamp_DIFF1_MIN'] = np.min(arr1)
#         featdict['behavior_timestamp_DIFF1_MAX'] = np.max(arr1)
#         featdict['behavior_timestamp_DIFF1_MEAN'] = np.mean(arr1)
#         featdict['behavior_timestamp_DIFF1_LEN'] = len(arr1)
        # featdict['behavior_timestamp_DIFF1_show_ratio'] = len(arr1) / (df[df['behavior_show'] == '1'].shape[0]+1)
        
#     if len(arr[arr > 600]) == 0:
#         featdict['behavior_timestamp_TIME_MAX'] = 0
#         featdict['behavior_timestamp_TIME_LEN'] = 0
#     else:
#         arr2= arr[arr > 600]
#         featdict['behavior_timestamp_TIME_MAX'] = np.max(arr2)
#         featdict['behavior_timestamp_TIME_LEN'] = len(arr2)
        
    featdict['behavior_timestamp_month_NUNIQUE'] = df['behavior_timestamp'].dt.month.nunique()
    featdict['behavior_timestamp_day_NUNIQUE'] = df['behavior_timestamp'].dt.day.nunique()
    featdict['behavior_timestamp_hour_NUNIQUE'] = df['behavior_timestamp'].dt.hour.nunique()
    featdict['behavior_timestamp_minute_NUNIQUE'] = df['behavior_timestamp'].dt.minute.nunique()
    
    featdict['behavior_timestamp_last_hour'] = df['behavior_timestamp'].iloc[-1].hour
    # featdict['behavior_timestamp_last_minute'] = df['behavior_timestamp'].iloc[-1].minute
    featdict['behavior_timestamp_last_time'] = df['behavior_timestamp'].iloc[-1].hour * 60 + df['behavior_timestamp'].iloc[-1].minute
    featdict['behavior_timestamp_last_time_2400'] = 24 * 60 -featdict['behavior_timestamp_last_time']
    
    featdict['behavior_timestamp_first_hour'] = 24 - df['behavior_timestamp'].iloc[0].hour
    featdict['behavior_timestamp_first_minute'] = df['behavior_timestamp'].iloc[0].minute
    featdict['behavior_timestamp_first_time'] = df['behavior_timestamp'].iloc[0].hour * 60 + df['behavior_timestamp'].iloc[0].minute
    featdict['behavior_timestamp_first_time_2400'] = 24 * 60 -featdict['behavior_timestamp_first_time']
    
#     if 'rec_type_1' in df['behavior_recommend'].values:
#         featdict['behavior_recommend_type1_last_hour'] = df[df['behavior_recommend'] == 'rec_type_1']['behavior_timestamp'].iloc[-1].hour
#     else:
#         featdict['behavior_recommend_type1_last_hour'] = 0    
#     if 'rec_type_2' in df['behavior_recommend'].values:
#         featdict['behavior_recommend_type2_last_hour'] = df[df['behavior_recommend'] == 'rec_type_2']['behavior_timestamp'].iloc[-1].hour
#     else:
#         featdict['behavior_recommend_type2_last_hour'] = 0
#     if 'rec_type_3' in df['behavior_recommend'].values:
#         featdict['behavior_recommend_type3_last_hour'] = df[df['behavior_recommend'] == 'rec_type_3']['behavior_timestamp'].iloc[-1].hour
#     else:
#         featdict['behavior_recommend_type3_last_hour'] = 0
    
    # 周中 信息编码
    for day in range(7):
        featdict['behavior_timestamp_weekday' + str(day)] = 0
    for day in df['behavior_timestamp'].dt.weekday.unique():
        featdict['behavior_timestamp_weekday' + str(day)] = 1
    
    featdict['behavior_timestamp_weekday_NUNIQUE'] = df['behavior_timestamp'].dt.weekday.nunique()
    
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
    featdict['behavior_timestamp_%minute'] = behavior_timestamp.seconds % 60
    featdict['behavior_timestamp_minute'] = behavior_timestamp.seconds / 60
    # featdict['behavior_timestamp_hour'] = behavior_timestamp.seconds % 3600
    # featdict['behavior_timestamp_%day'] = behavior_timestamp.seconds % 86400
    featdict['behavior_timestamp_day'] = behavior_timestamp.seconds / 86400
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

train_feat = Parallel(n_jobs=-1)(delayed(feature_agg)(i, './train/'+id+'.csv') for i, id in enumerate(train_id['user_id'].iloc[:]))
test_feat = Parallel(n_jobs=-1)(delayed(feature_agg)(i, './test/'+id+'.csv') for i, id in enumerate(test_id['user_id'].iloc[:]))

train_feat = pd.DataFrame(train_feat)
test_feat = pd.DataFrame(test_feat)

train_feat.to_hdf('train_test.hdf', 'train')
test_feat.to_hdf('train_test.hdf', 'test')

train_feat = pd.merge(train_feat, train_id, on='user_id', how='left')

params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'max_depth': -1,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'binary',
    'n_estimators': 4000,
    'metric': 'auc',
    # 'num_class': 6,
    'feature_fraction': .85,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 40,
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
    'num_threads': 40,
    'verbose': -1
}

train_pred, test_pred = run_oof(lgb.LGBMClassifier(**params), 
                                train_feat.drop(['user_id', 'label'], axis=1).values, 
                                train_feat['label'].values, 
                                test_feat.drop(['user_id'], axis=1).values, 
                                skf)

test_feat['label'] = test_pred
test_feat[['user_id', 'label']].to_csv('baseline.csv', index=None, header=None)