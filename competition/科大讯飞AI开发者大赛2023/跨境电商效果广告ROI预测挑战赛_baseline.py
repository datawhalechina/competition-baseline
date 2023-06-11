import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('跨境电商效果广告ROI预测挑战赛公开数据/train.csv')
test_data = pd.read_csv('跨境电商效果广告ROI预测挑战赛公开数据/testA.csv')

train_data['datetime'] = pd.to_datetime(train_data['datetime'])
test_data['datetime'] = pd.to_datetime(test_data['datetime'])
train_data['datetime_hour'] = train_data['datetime'].dt.hour
test_data['datetime_hour'] = test_data['datetime'].dt.hour

train_data.drop('datetime', axis=1, inplace=True)
test_data.drop('datetime', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

for col in ['ad_id', 'ad_set_id', 'campaign_id', 'product_id', 'account_id', 'post_id_emb', 'post_type', 'countries']:
    lbl = LabelEncoder()
    lbl.fit(list(train_data[col]) + list(test_data[col]))
    train_data[col] = lbl.transform(list(train_data[col]))
    test_data[col] = lbl.transform(list(test_data[col]))

from lightgbm import LGBMRegressor
model = LGBMRegressor()

train_data['product_id_roi_mean'] = train_data['product_id'].map(train_data.groupby(['product_id'])['roi'].mean())
test_data['product_id_roi_mean'] = test_data['product_id'].map(train_data.groupby(['product_id'])['roi'].mean())

train_data['account_id_roi_mean'] = train_data['account_id'].map(train_data.groupby(['account_id'])['roi'].mean())
test_data['account_id_roi_mean'] = test_data['account_id'].map(train_data.groupby(['account_id'])['roi'].mean())

train_data['countries_roi_mean'] = train_data['countries'].map(train_data.groupby(['countries'])['roi'].mean())
test_data['countries_roi_mean'] = test_data['countries'].map(train_data.groupby(['countries'])['roi'].mean())

train_data['datetime_hour_roi_mean'] = train_data['datetime_hour'].map(train_data.groupby(['datetime_hour'])['roi'].mean())
test_data['datetime_hour_roi_mean'] = test_data['datetime_hour'].map(train_data.groupby(['datetime_hour'])['roi'].mean())

model.fit(
    train_data.iloc[:].drop('roi', axis=1),
    train_data.iloc[:]['roi'], categorical_feature=['ad_id', 'ad_set_id', 'campaign_id', 'product_id', 'account_id', 'post_id_emb', 'post_type', 'countries']
)

df = pd.read_csv('提交示例.csv')
df['roi'] = model.predict(test_data.iloc[:].drop('uuid', axis=1))
df.to_csv('submit.csv', index=None)
