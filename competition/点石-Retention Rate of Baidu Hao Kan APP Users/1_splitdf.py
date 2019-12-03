import pandas as pd
import numpy as np

import os, sys, time, codecs, glob
from tqdm import tqdm, tqdm_notebook

def read_input(debug=True):
    if debug:
        nrows = 100000
    else:
        nrows = None

    train = pd.read_csv('../input/train', sep='\t', nrows=nrows,
                names=['user_id', 'user_male', 'user_age', 'user_edu', 'user_district', 'label',  'user_install',
                        'video_id', 'video_class', 'video_tag', 'video_creator', 'video_uptime', 'video_duration',
                        'behavior_show', 'behavior_click', 'behavior_recommend', 'behavior_playback', 'behavior_timestamp',
                        'behavior_comment', 'behavior_like', 'behavior_forard'],
                dtype={'user_id':object, 'video_tag':object})
    test = pd.read_csv('../input/test', sep='\t', nrows=nrows,
                names=['user_id', 'user_male', 'user_age', 'user_edu', 'user_district',  'user_install',
                        'video_id', 'video_class', 'video_tag', 'video_creator', 'video_uptime', 'video_duration',
                        'behavior_show', 'behavior_click', 'behavior_recommend', 'behavior_playback', 'behavior_timestamp',
                        'behavior_comment', 'behavior_like', 'behavior_forard'])

#     train['video_uptime'] = train['video_uptime'].apply(lambda x: timestamp_datetime(x))
#     train['behavior_timestamp'] = train['behavior_timestamp'].apply(lambda x: timestamp_datetime(x / 1000))
#     train['video_tag'] = train['video_tag'].apply(lambda x: x.split('$'))
#     train.sort_values(by=['user_id', 'behavior_timestamp'], inplace=True)


#     test['video_uptime'] = test['video_uptime'].apply(lambda x: timestamp_datetime(x))
#     test['behavior_timestamp'] = test['behavior_timestamp'].apply(lambda x: timestamp_datetime(x / 1000))
#     test['video_tag'] = test['video_tag'].apply(lambda x: x.split('$'))
#     test.sort_values(by=['user_id', 'behavior_timestamp'], inplace=True)

    return train, test

train, test = read_input(debug=False)

# idx = train['user_id'].value_counts()
# idx = idx[train['user_id'].unique()]
# idx = idx.reset_index()
# for i, rows in tqdm(enumerate(idx.iterrows())):
#     if i == 0:
#         start = 0
#     else:
#         start = idx.iloc[:i]['user_id'].sum()
#     span = idx.iloc[i]['user_id']

#     tmp_df = train.iloc[start :start+span]
#     tmp_df.to_csv('./train/{0}.csv'.format(str(idx.iloc[i]['index'])), index=None)

idx = test['user_id'].value_counts()
idx = idx[test['user_id'].unique()]
idx = idx.reset_index()
for i, rows in tqdm(enumerate(idx.iterrows())):
    if i == 0:
        start = 0
    else:
        start = idx.iloc[:i]['user_id'].sum()
    span = idx.iloc[i]['user_id']

    tmp_df = test.iloc[start :start+span]
    tmp_df.to_csv('./test/{0}.csv'.format(str(idx.iloc[i]['index'])), index=None)
