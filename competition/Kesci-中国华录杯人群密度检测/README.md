比赛链接：https://www.kesci.com/home/competition/5df1d33d23ea6d002b264ada/content

人群密度检测：在一张图片当中统计图片当中行人的数量。特别说明，当画面中行人数量大于 100 时，均按 100 计算。

![](https://github.com/weizheliu/Context-Aware-Crowd-Counting/raw/master/images/prediction.png)

比赛数据集链接：链接: https://pan.baidu.com/s/1wtmQUlsr_fcUKGTW1K-4oA 提取码: c2ab

baseline思路，使用Crowd Counting进行预测，使用*Context-Aware Crowd Counting*的预训练权重：

1. `git clone https://github.com/weizheliu/Context-Aware-Crowd-Counting`
2. 下载pretrained model（part_B_pre.pth.tar），在我们分享的数据集中已经包含
3. `python test.py`即可，线上分数341左右
