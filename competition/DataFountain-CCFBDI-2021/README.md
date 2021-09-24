## 个贷违约预测

- 赛题类型：结构化数据挖掘、金融风控

https://www.datafountain.cn/competitions/530

本赛题要求利用已有的与目标客群稍有差异的另一批信贷数据，辅助目标业务风控模型的创建，两者数据集之间存在大量相同的字段和极少的共同用户。此处希望大家可以利用迁移学习捕捉不同业务中用户基本信息与违约行为之间的关联，帮助实现对新业务的用户违约预测。

- baseline1：[阿水0.86单表思路](https://github.com/datawhalechina/competition-baseline/blob/master/competition/DataFountain-CCFBDI-2021/%E4%B8%AA%E8%B4%B7%E8%BF%9D%E7%BA%A6%E9%A2%84%E6%B5%8B-860.ipynb)
- baseline2：[恒哥0.87多表思路](https://github.com/LogicJake/competition_baselines/tree/master/competitions/2021ccf_loan)

## 剧本角色情感识别

- 赛题类型：NLP、情感分类

https://www.datafountain.cn/competitions/518

本赛题提供一部分电影剧本作为训练集，训练集数据已由人工进行标注，参赛队伍需要对剧本场景中每句对白和动作描述中涉及到的每个角色的情感从多个维度进行分析和识别。该任务的主要难点和挑战包括：1）剧本的行文风格和通常的新闻类语料差别较大，更加口语化；2）剧本中角色情感不仅仅取决于当前的文本，对前文语义可能有深度依赖。

- basline1：[恒哥 Bert 0.682](https://github.com/LogicJake/competition_baselines/tree/master/competitions/2021ccf_aqy)
- basline2：[强哥 Bert多任务 0.67](https://github.com/China-ChallengeHub/ChallengeHub-Baselines/blob/main/aiqiyi-baseline.ipynb)

![](https://coggle.club/assets/img/coggle_qrcode.jpg)


## 用户上网异常行为分析

- 赛题类型：结构化数据挖掘

https://www.datafountain.cn/competitions/520

利用机器学习、深度学习，UEBA等人工智能方法，基于无标签的用户日常上网日志数据，构建用户上网行为基线和上网行为评价模型，依据上网行为与基线的距离确定偏离程度。
- 通过用户日常上网数据构建行为基线；
- 采用无监督学习模型，基于用户上网行为特征，构建上网行为评价模型，评价上网行为与基线的偏离程度。

- baseline：[CquptDJ](https://blog.csdn.net/qq_44694861/article/details/120423658)
