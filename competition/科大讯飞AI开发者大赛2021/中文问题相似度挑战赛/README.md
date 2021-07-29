## 中文问题相似度挑战赛

### 赛事背景
问答系统中包括三个主要的部分：问题理解，信息检索和答案抽取。而问题理解是问答系统的第一部分也是非常关键的一部分。问题理解有非常广泛的应用，如重复评论识别、相似问题识别等。

重复问题检测是一个常见的文本挖掘任务，在很多实际问答社区都有相应的应用。重复问题检测可以方便进行问题的答案聚合，以及问题答案推荐，自动QA等。由于中文词语的多样性和灵活性，本赛题需要选手构建一个重复问题识别算法。

### 赛事任务
本次赛题希望参赛选手对两个问题完成相似度打分。

训练集：约5千条问题对和标签。若两个问题是相同的问题，标签为1；否则为0。

测试集：约5千条问题对，需要选手预测标签。

http://challenge.xfyun.cn/topic/info?type=chinese-question-similarity&ch=dw-sq-1

### baseline

- [BERT NSP方法](https://github.com/datawhalechina/competition-baseline/blob/master/competition/%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9EAI%E5%BC%80%E5%8F%91%E8%80%85%E5%A4%A7%E8%B5%9B2021/%E4%B8%AD%E6%96%87%E9%97%AE%E9%A2%98%E7%9B%B8%E4%BC%BC%E5%BA%A6%E6%8C%91%E6%88%98%E8%B5%9B/bert-nsp.ipynb)
- [word2vec + LightGBM](https://mp.weixin.qq.com/s/E3sfNaNg8JH-w_7Yv40MWw), 链接：https://pan.baidu.com/s/1WC3vQGlgBFvnlAXcj-0qrA 提取码：v7aj 
