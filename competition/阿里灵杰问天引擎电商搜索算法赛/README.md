![](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/164611999657774301646119988943.png)


## [https://coggle.club/blog/tianchi-open-search](https://coggle.club/blog/tianchi-open-search)

## 比赛介绍

受疫情催化影响，近一年内全球电商及在线零售行业进入高速发展期。作为线上交易场景的重要购买入口，搜索行为背后是强烈的购买意愿，电商搜索质量的高低将直接决定最终的成交结果，因此在AI时代，如何通过构建智能搜索能力提升线上GMV转化成为了众多电商开发者的重要研究课题。**本次比赛由阿里云天池平台和问天引擎联合举办，诚邀社会各界开发者参与竞赛，共建AI未来。**


## 赛题建模

赛题是一个文本检索任务：给定一个搜索查询，我们首先使用一个检索系统来检索得结果。但检索系统可能会检索与搜索查询不相关的文档，整体的任务可以参考已有的`文本语义检索`。

![](https://cdn.coggle.club/img/InformationRetrieval.png)

### 赛题数据分析

- 文本长度分析

- 关键词分析

- hard example

### 赛题难点分析

赛题的query比较短，属于非对称语义搜索（Asymmetric Semantic Search）任务，有一个简短的查询，希望找到一个较长的段落来回答该查询。赛题的query与corpus的文本可能存在并无重合单词的情况。

![](https://cdn.coggle.club/img/SemanticSearch.png)

### 赛题解题思路

- 思路1：使用关键词匹配，识别出query和corpus中关键词，使用关键词进行编码为向量。
- 思路2：使用sentence-bert结合比赛标注数据进行训练
- 思路3：使用simcse无监督对比学习训练

### 赛题相关资料

- [https://www.sbert.net/examples/training/data_augmentation/README.html](https://www.sbert.net/examples/training/data_augmentation/README.html)
- [https://www.sbert.net/examples/applications/semantic-search/README.html](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [https://www.sbert.net/docs/pretrained-models/msmarco-v3.html](https://www.sbert.net/docs/pretrained-models/msmarco-v3.html)
