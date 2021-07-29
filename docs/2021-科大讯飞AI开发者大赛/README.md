### Part1 讯飞赛事介绍

2021年A.I.开发者大赛继续秉承 “技术顶天、应用立地”的坚定理念，开放科大讯飞海量数据资源及人工智能核心技术，全面升级A.I.算法赛、A.I.应用赛、A.I.公益赛、A.I.缤纷赛四大赛道，面向全球开发者，激发人工智能多个行业领域应用的创新探索与挑战。

算法赛与应用赛全方位覆盖了智能语音、CV、NLP、OCR、AR人机交互等人工智能热门研究，同时深耕农业养殖、生物与环保、医疗健康、地理遥感、企业数字化、新能源汽车、金融信息化、智慧城市等多领域多行业方向，期待开发者们尽情展示算法与应用的智慧演练！
除了面向全球专业开发者的数据算法及创新应用两大经典赛道，为进一步赋能行业与生活场景，2021 iFLYTEK A.I.开发者大赛针对赛道进行了创新性升级，丰富的赛题内容带给选手更多的可能性，下面就让我们看看今年都有哪些赛题吧！


### Part2 学术论文分类挑战赛

- 赛题类型：自然语言处理
- 赛题任务：文本分类
- 赛题链接：http://challenge.xfyun.cn/topic/info?type=academic-paper-classification&ch=dw-sq-1

#### 赛题背景 & 任务

随着人工智能技术不断发展，每周都有非常多的论文公开发布。现如今对论文进行分类逐渐成为非常现实的问题，这也是研究人员和研究机构每天都面临的问题。现在希望选手能构建一个论文分类模型。

本次赛题希望参赛选手利用论文信息：论文id、标题、摘要，划分论文具体类别。

```
paperid：9821
title：Calculation of prompt diphoton production cross sections at Tevatron and LHC energies
abstract：A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy.
categories：hep-ph
```

训练数据和测试集以csv文件给出，其中：
- 训练集5W篇论文。其中每篇论文都包含论文id、标题、摘要和类别四个字段。
- 测试集1W篇论文。其中每篇论文都包含论文id、标题、摘要，不包含论文类别字段。

本次竞赛的评价标准采用准确率指标，最高分为1。计算方法参考https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html， 评估代码参考：

```
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
```

#### 赛题解题思路

赛题是一个典型的文本分类任务，所以可以使用文本分类的思路来完成。在本赛题我们主要尝试两个思路：
- 方法1：文本TFIDF特征抽取 + 线性分类
- 方法2：Bert模型文本分类

##### 方法1：文本TFIDF特征抽取 + 线性分类

TFIDF是非常常见的文本特征提取方法，使用的方法非常简单，可以直接借助sklearn来完成TFIDF特征提取。

##### 方法2：Bert模型文本分类

### Part3 中文问题相似度挑战赛

http://challenge.xfyun.cn/topic/info?type=chinese-question-similarity&ch=dw-sq-1

#### 赛事背景 & 任务

问答系统中包括三个主要的部分：问题理解，信息检索和答案抽取。而问题理解是问答系统的第一部分也是非常关键的一部分。问题理解有非常广泛的应用，如重复评论识别、相似问题识别等。

重复问题检测是一个常见的文本挖掘任务，在很多实际问答社区都有相应的应用。重复问题检测可以方便进行问题的答案聚合，以及问题答案推荐，自动QA等。由于中文词语的多样性和灵活性，本赛题需要选手构建一个重复问题识别算法。

本次赛题希望参赛选手对两个问题完成相似度打分。

- 训练集：约5千条问题对和标签。若两个问题是相同的问题，标签为1；否则为0。
- 测试集：约5千条问题对，需要选手预测标签。

#### 赛题解题思路

赛题是一个典型的文本匹配任务，可以使用文本匹配的思路完成。在本赛题我们主要尝试两个思路：
- 方法1：文本相似度 + 树模型分类
- 方法2：Bert NSP任务