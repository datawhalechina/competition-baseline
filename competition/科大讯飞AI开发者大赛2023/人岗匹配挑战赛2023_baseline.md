本地目录如下：

```
person-post-matching-2023/
  run.py
  train.json    从比赛官网下载
  job_list.json   从比赛官网下载
```

打包提交过程
```
tar -cvzf person-post-matching-2023.tar.gz person-post-matching-2023/
s3cmd put person-post-matching-2023.tar.gz s3://ai-competition/你的url/
```

run.py代码内容如下：

```python
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

train_data = pd.read_json('./train.json')
train_data['解析结果'] = train_data['解析结果'].apply(lambda x : json.dumps(x).replace('"', ' ').replace('"', ' ').split())

test_data = pd.read_json('/work/data/personnel-matching-test-set/test.json')
test_data['解析结果'] = test_data['解析结果'].apply(lambda x : json.dumps(x).replace('"', ' ').replace('"', ' ').split())

joblist = pd.read_json('./job_list.json')
joblist['解析结果'] = joblist['岗位名称'] + ' ' + joblist['岗位介绍'] + ' ' + joblist['岗位要求']
joblist['解析结果'] = joblist['解析结果'].apply(lambda x : x.split())

train_feat = []
for row in train_data.iterrows():
    label = row[1]['岗位ID']
    query_text= row[1]['解析结果']
    feat = [
        label,
        len(query_text), len(set(query_text)), len(query_text) - len(set(query_text)),
    ]
    for target_text in joblist['解析结果']:
        feat += [
            len(set(query_text) & set(target_text)),
            len(set(query_text) & set(target_text)) / len(query_text),
            len(set(query_text) & set(target_text)) / len(target_text),
            
            len(set(query_text) & set(target_text)) / len(set(target_text)),
            len(set(query_text) & set(target_text)) / len(set(query_text))

        ]
    train_feat.append(feat)
train_feat = np.array(train_feat)
m = RandomForestClassifier()
m.fit(
    train_feat[:, 1:],
    train_feat[:, 0],
)

test_feat = []
for row in test_data.iterrows():
    query_text= row[1]['解析结果']
    feat = [
        len(query_text), len(set(query_text)), len(query_text) - len(set(query_text)),
    ]
    for target_text in joblist['解析结果']:
        feat += [
            len(set(query_text) & set(target_text)),
            len(set(query_text) & set(target_text)) / len(query_text),
            len(set(query_text) & set(target_text)) / len(target_text),
            
            len(set(query_text) & set(target_text)) / len(set(target_text)),
            len(set(query_text) & set(target_text)) / len(set(query_text))

        ]
    test_feat.append(feat)
test_feat = np.array(test_feat)
pd.DataFrame({
    '简历ID': range(len(test_data)),
    '岗位ID': m.predict(test_feat).astype(int)
}).to_csv('/work/output/result.csv', index=None)

```
