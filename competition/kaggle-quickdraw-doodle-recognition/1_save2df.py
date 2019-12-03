#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, codecs, glob
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# 读取单个csv文件
def read_df(path, nrows):
    print('Reading...', path)
    if nrows.isdigit():
        return pd.read_csv(path, nrows=int(nrows), parse_dates=['timestamp'])
    else:
        return pd.read_csv(path, parse_dates=['timestamp'])

# 读取多个csv文件
def contcat_df(paths, nrows):
    dfs = []
    for path in paths:
        dfs.append(read_df(path, nrows))
    return pd.concat(dfs, axis=0, ignore_index=True)

def main():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    
    CLASSES_CSV = glob.glob('../input/train_simplified/*.csv')
    CLASSES = [x.split('/')[-1][:-4] for x in CLASSES_CSV]

    print('Reading data...')
    df = contcat_df(CLASSES_CSV, number)
    df = df.reindex(np.random.permutation(df.index))
    
    lbl = LabelEncoder().fit(df['word'])
    df['word'] = lbl.transform(df['word'])
    
    if df.shape[0] * 0.05 < 120000:
        df_train, df_val = train_test_split(df, test_size=0.05)
    else:
        df_train, df_val = df.iloc[:-500000], df.iloc[-500000:]
    
    print('Train:', df_train.shape[0], 'Val', df_val.shape[0])
    print('Save data...')
    df_train.to_pickle(os.path.join('./data/', 'train_' + str(number) + '.pkl'))
    df_val.to_pickle(os.path.join('./data/', 'val_' + str(number) + '.pkl'))

# python 1_save2df.py 50000
# python 1_save2df.py all
if __name__ == "__main__":
    number = str(sys.argv[1])
    main()