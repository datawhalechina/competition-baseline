#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:quincy qiang
# email:yanqiangmiffy@gmail.com
# datetime:2021/8/16 11:20
# description:"do something"

import re
import pandas as pd
from tqdm import tqdm

train = pd.read_csv('data/train.csv', sep='\t')
test = pd.read_csv('data/test.csv', sep='\t')

print(train)
print(test)


def process_text(text):
    return re.sub(' +', ' ', text).strip()


def get_question(text):
    """
    根据[MASK][MASK][MASK][MASK]获取问题
    :param text:
    :return:
    """
    sentences = re.split('(。|！|\!|\.|？|\?)', text)  # 保留分割符
    for sent in sentences:
        if '[MASK][MASK][MASK][MASK]' in sent:
            return sent
    return text


cols = [
    "Unnamed: 0",
    "video-id",
    "fold-ind",  # q_id
    "startphrase",
    "sent1",  # content
    "sent2",  # question
    "gold-source",
    "ending0", "ending1", "ending2", "ending3",  # choice
    "label"]

# ======================================================
# 生成训练集
# ======================================================
res = []

for idx, row in tqdm(train.iterrows()):
    q_id = f'train_{idx}'
    content = row['text']
    content = process_text(content)
    question = get_question(content)
    modified_choices = eval(row['candidate'])
    label = modified_choices.index(row['label'])
    ## Hard-code for swag format!
    res.append(("",
                "",
                q_id,
                "",
                content,
                question,
                "",
                modified_choices[0],
                modified_choices[1],
                modified_choices[2],
                modified_choices[3],
                label))
df = pd.DataFrame(res, columns=cols)

# ======================================================
# 生成测试集
# ======================================================
res = []
print("test.shape", test.shape)
for idx, row in tqdm(test.iterrows()):
    q_id = f'test_{idx}'
    content = row['text']
    content = process_text(content)
    question = get_question(content)
    modified_choices = eval(row['candidate'])
    ## Hard-code for swag format!
    res.append(("",
                "",
                q_id,
                "",
                content,
                question,
                "",
                modified_choices[0],
                modified_choices[1],
                modified_choices[2],
                modified_choices[3],
                0))
df_test = pd.DataFrame(res, columns=cols)

print(df_test.shape)


DEBUG = False
if DEBUG:
    df.iloc[:50].to_csv('data/new_train.csv', index=False)
    df.iloc[-50:].to_csv('data/new_valid.csv', index=False)
    df_test.iloc[:50].to_csv('data/new_test.csv', index=False)
else:
    df.iloc[:45000].to_csv('data/new_train.csv', index=False)
    df.iloc[5000:].to_csv('data/new_valid.csv', index=False)
    df_test.to_csv('data/new_test.csv', index=False)
