import pandas as pd # 读取文件
import numpy as np # 数值计算
import nagisa # 日文分词
from sklearn.feature_extraction.text import TfidfVectorizer # 文本特征提取
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.pipeline import make_pipeline # 组合流水线

# 读取数据
train_cn = pd.read_excel('汽车领域多语种迁移学习挑战赛初赛训练集/中文_trian.xlsx')
train_ja = pd.read_excel('汽车领域多语种迁移学习挑战赛初赛训练集/日语_train.xlsx')
train_en = pd.read_excel('汽车领域多语种迁移学习挑战赛初赛训练集/英文_train.xlsx')

test_ja = pd.read_excel('testA.xlsx', sheet_name='日语_testA')
test_en = pd.read_excel('testA.xlsx', sheet_name='英文_testA')

# 文本分词
train_ja['words'] = train_ja['原始文本'].apply(lambda x: ' '.join(nagisa.tagging(x).words))
train_en['words'] = train_en['原始文本'].apply(lambda x: x.lower())

test_ja['words'] = test_ja['原始文本'].apply(lambda x: ' '.join(nagisa.tagging(x).words))
test_en['words'] = test_en['原始文本'].apply(lambda x: x.lower())

# 训练TFIDF和逻辑回归
pipline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)
pipline.fit(
    train_ja['words'].tolist() + train_en['words'].tolist(),
    train_ja['意图'].tolist() + train_en['意图'].tolist()
)

# 模型预测
test_ja['意图'] = pipline.predict(test_ja['words'])
test_en['意图'] = pipline.predict(test_en['words'])
test_en['槽值1'] = np.nan
test_en['槽值2'] = np.nan

test_ja['槽值1'] = np.nan
test_ja['槽值2'] = np.nan

# 写入提交文件
writer = pd.ExcelWriter('submit.xlsx')
test_en.drop(['words'], axis=1).to_excel(writer, sheet_name='英文_testA', index=None)
test_ja.drop(['words'], axis=1).to_excel(writer, sheet_name='日语_testA', index=None)
writer.save()
writer.close()
