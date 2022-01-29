**WSDM会议（CCF B类会议）** 是涉及搜索和数据挖掘的网络启发研究的主要会议之一。WSDM Cup将于10月15日开始，一直持续2022年到1月下旬。

> 比赛赛题解析录屏（11月28日）：https://www.bilibili.com/video/BV1Ng411K7Jm/

## User Retention Score Prediction

http://challenge.ai.iqiyi.com/detail?raceId=61600f6cef1b65639cd5eaa6

举办方：iQIYI

赛题类型：用户留存预测、CTR类型

### 赛题背景

爱奇艺手机端APP，通过深度学习等最新的AI技术，提升用户个性化的产品体验，更好地让用户享受定制化的娱乐服务。我们用“N日留存分”这一关键指标来衡量用户的满意程度。

例如，如果一个用户10月1日的“7日留存分”等于3，代表这个用户接下来的7天里（10月2日~8日），有3天会访问爱奇艺APP。预测用户的留存分是个充满挑战的难题：不同用户本身的偏好、活跃度差异很大，另外用户可支配的娱乐时间、热门内容的流行趋势等其他因素，也有很强的周期性特征。

### 赛题任务

本次大赛基于爱奇艺APP脱敏和采样后的数据信息，预测用户的7日留存分。参赛队伍需要设计相应的算法进行数据分析和预测。

### 评价指标
本次比赛是一个数值预测类问题。评价函数使用：$100*(1-\frac{1}{n}\sum^n_1|\frac{F_t-A_t}{7}|)$

$n$是测试集用户数量，$F$是参赛者对用户的7日留存分预测值，$A$是真实的7日留存分真实值。

### 赛题开源

- [第一名思路](https://zhuanlan.zhihu.com/p/462736790), [代码](https://github.com/hansu1017/WSDM2022-Retention-Score-Prediction)
- [第三名代码](https://github.com/Chenfei-Kang/2022_WSDM_iQiYi_Retention_Score_Prediction)

### 其他开源

- [`举办方`开源了84.5分数的代码](http://challenge.ai.iqiyi.com/detail?raceId=61600f6cef1b65639cd5eaa6)，基于Keras，需要32G内存 + 4G GPU
- [`阿水`基于举办方改写了模型代码](https://aistudio.baidu.com/aistudio/projectdetail/2715522)，线上85.5，基于PaddlePaddle，需要32G内存 + 4G GPU
- [`第一次打比赛`只使用了两个特征](https://github.com/LogicJake/competition_baselines/tree/master/competitions/wsdm_iqiyi_torch)，基于Pytorch，需要8G内存 + 4G GPU

## Temporal Link Prediction

https://www.dgl.ai/WSDM2022-Challenge/

举办方：Intel / Amazon

比赛类型：图算法

### 赛题背景

Temporal Link Prediction是时间图上的经典任务之一。与询问部分观察图上两个节点之间是否存在边的链接预测相反，时间链接预测询问在给定时间跨度内两个节点之间是否存在边。

它比传统的链接预测更有用，因为可以围绕模型构建多个应用程序，例如预测电子商务中客户的需求，或预测社交网络中将发生什么事件等。

### 赛题任务

在这个挑战中，我们希望有一个模型可以同时处理两种数据：

- 数据集 A：以实体为节点，以不同类型的事件为边的动态事件图。
- 数据集 B：用户-项目图，以用户和项目为节点，以不同类型的交互为边。

该任务将预测在给定时间戳之前两个给定节点之间是否存在给定类型的边。


### 评价指标

使用 ROC 下的面积 (AUC) 作为两个数据集的评估指标，并使用两个$AUC$的调和平均值作为提交的分数。

具体来说设$AUC_A$和$AUC_B$分别为数据集A和数据集B的$AUC$。

## Cross- Market Recommendation

https://xmrec.github.io/wsdmcup/

举办方：University of Amsterdam / University of Massachusetts Amherst / Amazon

比赛类型：推荐系统

### 赛题背景

电子商务公司通常跨市场运营；例如亚马逊已将业务和销售扩展到全球18 个市场（即国家/地区）。跨市场推荐涉及通过利用类似的高资源市场的数据向目标市场的用户推荐相关产品的问题，例如利用美国市场的数据改进目标市场的推荐。

然而关键的挑战是数据，例如用户与产品的交互数据（点击、购买、评论），传达了个别市场的某些偏见。因此在源市场上训练的算法在不同的目标市场不一定有效。

### 赛题目标

在本次WSDM杯挑战赛中，我们提供不同市场的用户购买和评分数据，目标是通过利用来自类似辅助市场的数据来改进这些目标市场中的个人推荐系统。

### 评估指标

使用NDCG@10进行评估，项目的分数为每个用户排序，前10个项目被考虑进行评估。



