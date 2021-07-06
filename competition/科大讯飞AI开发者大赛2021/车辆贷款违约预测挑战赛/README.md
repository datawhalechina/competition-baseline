# 车辆贷款违约预测挑战赛baseline

本文旨在帮助各位选手快速入门比赛，搭建一个可出结果的baseline代码，另外也会给出相关知识点和学习方向，帮助快速提升成绩和竞赛入门。

比赛地址：http://challenge.xfyun.cn/topic/info?type=car-loan

- 本文旨在帮助各位选手快速入门比赛，搭建一个可出结果的baseline代码，另外也会给出相关知识点和学习方向，帮助快速提升成绩和竞赛入门。

## 1.导入第三方包

```python
import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')
```

## 2.读取数据

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

sample_submit = pd.read_csv('sample_submit.csv')
```

## 3.训练数据/测试数据准备

```python
all_cols = [f for f in train.columns if f not in ['customer_id','loan_default']]

x_train = train[all_cols]
x_test = test[all_cols]

y_train = train['loan_default']
```

#### 特征工程

在训练数据/测试数据准备部分仅使用的原始特征，并没有进行过多的特征工程相关工作，所以这里还是很值得优化的，并且相信提升点是非常多的。下面介绍特征工程中需要做的事情。

- 特征工程对比赛结果的影响非常大哦，这里给出关于特征交互、特征编码和特征选择的介绍。
- 特征交互

交互特征的构造非常简单，使用起来却代价不菲。如果线性模型中包含有交互特征对，那它的训练时间和评分时间就会从 O(n) 增加到 O(n2)，其中 n 是单一特征的数量。

- - 特征和特征之间组合
  - 特征和特征之间衍生

- 特征编码

- - one-hot编码
  - label-encode编码

- 特征选择

特征选择技术可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度。特征选择不是为了减少训练时间（实际上，一些技术会增加总体训练时间），而是为了减少模型评分时间

- - 1 Filter

  - - VarianceThreshold(threshold=3).fit_transform(train,target_train)
    - SelectKBest(k=5).fit_transform(train,target_train)
    - SelectKBest(chi2, k=5).fit_transform(train,target_train)

  - 2 Wrapper （RFE）

  - - RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(train,target_train)

  - 3 Embedded

  - - SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(train,target_train)
    - SelectFromModel(GradientBoostingClassifier()).fit_transform(train,target_train)

## 4.模型训练

作为baseline部分仅使用经典的**LightGBM**作为训练模型，我们还能尝试**XGBoost、CatBoost和NN（神经网络）**

- **上分利器**

- - XGBoost模型

  - - https://blog.csdn.net/wuzhongqiang/article/details/104854890

  - LightGBM模型

  - - https://blog.csdn.net/wuzhongqiang/article/details/105350579

```python
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
```

```python
lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test)
```

#### 参数优化

对于模型的参数部分baseline部分并没有进行过多的优化和实验，当然这也是个比较大的优化的，下面给出参考方法。

- 模型调参：

- - 贪心调参方法；

  - - 先使用当前对模型影响最大的参数进行调优，达到当前参数下的模型最优化，再使用对模型影响次之的参数进行调优，如此下去，直到所有的参数调整完毕。

    - 常用的参数和调参顺序：

    - - ①：max_depth、num_leaves
      - ②：min_data_in_leaf、min_child_weight
      - ③：bagging_fraction、 feature_fraction、bagging_freq
      - ④：reg_lambda、reg_alpha
      - ⑤：min_split_gain

  - 网格调参方法；

  - - sklearn 提供GridSearchCV用于进行网格搜索，只需要把模型的参数输进去，就能给出最优化的结果和参数。相比起贪心调参，网格搜索的结果会更优，但是网格搜索只适合于小数据集，一旦数据的量级上去了，很难得出结果。

  - 贝叶斯调参方法；

  - - 给定优化的目标函数(广义的函数，只需指定输入和输出即可，无需知道内部结构以及数学性质)，通过不断地添加样本点来更新目标函数的后验分布(高斯过程,直到后验分布基本贴合于真实分布）。简单的说，就是考虑了上一次参数的信息，从而更好的调整当前的参数。

## 5.搜索最佳阈值

- 因为使用F1评价指标，所以需要将概率结果转为整数，这里就需要转化阈值，即大于这个值的为1，小于等于这个值的为0，可以使用贪心的方式进行搜索。此部分配套Baseline中未加入，感兴趣的小伙伴可自行尝试。

```python
for thr in [0.2,0.25,0.3,0.35,0.4]:
    y_true = y_train
    y_pred = pd.DataFrame(lgb_train)[0].apply(lambda x:1 if x>thr else 0).values
    score = f1_score(y_true, y_pred, average='macro')
    print(thr, score)
```

```
0.2 0.5650153119932877
0.25 0.5832478807323747
0.3 0.5555528974982635
0.35 0.5161158663127036
0.4 0.4839455321941454
0.45 0.4672067583360815
0.5 0.45872038155379824
0.55 0.45441224134973596
0.6 0.4526113642887671
0.65 0.45180737509767505
```

可以看到0.25为最佳转化阈值，所以最终提交结果也适用这个阈值。

## 6.预测结果

```python
sample_submit['loan_default'] = lgb_test
sample_submit['loan_default'] = sample_submit['loan_default'].apply(lambda x:1 if x>0.25 else 0).values
sample_submit.to_csv('baseline_result', index=False)
```

## 7.更多学习方向

#### 数据分析

- 数据总体了解：

- - 通过info熟悉数据类型；

  - - data_train.info()

  - 粗略查看数据集中各特征基本统计量；

  - - data_train.describe()
    - data_train.head(3).append(data_train.tail(3))

- 缺失值和唯一值：

- - 查看数据缺失值情况

  - - data_train.isnull().any().sum()#有多少列有空值

    - - 纵向了解哪些列存在 “nan”, 并可以把nan的个数打印，主要的目的在于查看某一列nan存在的个数是否真的很大，如果nan存在的过多，说明这一列对label的影响几乎不起作用了，可以考虑删掉。如果缺失值很小一般可以选择填充。

    - 另外可以横向比较，如果在数据集中，某些样本数据的大部分列都是缺失的且样本足够的情况下可以考虑删除。

- 深入数据-查看数据类型

- - 特征一般都是由类别型特征和数值型特征组成，而数值型特征又分为连续型和离散型。
  - 类别型特征有时具有非数值关系，有时也具有数值关系。比如‘grade’中的等级A，B，C等，是否只是单纯的分类，还是A优于其他要结合业务判断。

- 类别型数据&数值型数据初步划分

- - numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
  - category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))

- 用pandas_profiling生成数据报告

- - 帮助更好了解数据情况
  - pandas_profiling.ProfileReport(data_train)

 #### 模型融合

**基础上分**

简单平均和加权平均是常用的两种比赛中模型融合的方式。其优点是快速、简单。

- 平均：（简单实用）

- - 简单平均法
  - 加权平均法

- 投票：

- - 简单投票法
  - 加权投票法

- 综合：

- - 排序融合
  - log融合

**进阶上分**

stacking在众多比赛中大杀四方，但是跑过代码的小伙伴想必能感受到速度之慢，同时stacking多层提升幅度并不能抵消其带来的时间和内存消耗，所以实际环境中应用还是有一定的难度，同时在有答辩环节的比赛中，主办方也会一定程度上考虑模型的复杂程度，所以说并不是模型融合的层数越多越好的。

- stacking:

- - 构建多层模型，并利用预测结果再拟合预测。

- blending：

- - 选取部分数据预测训练得到预测结果作为新特征，带入剩下的数据中预测。

- boosting/bagging

当然在比赛中将加权平均、stacking、blending等混用也是一种策略，可能会收获意想不到的效果哦！
