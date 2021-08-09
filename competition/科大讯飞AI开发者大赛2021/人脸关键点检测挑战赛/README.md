http://challenge.xfyun.cn/topic/info?type=key-points-of-human-face&ch=dw-sq-1

## 赛事背景
人脸识别是基于人的面部特征信息进行身份识别的一种生物识别技术，金融和安防是目前人脸识别应用最广泛的两个领域。人脸关键点是人脸识别中的关键技术。人脸关键点检测需要识别出人脸的指定位置坐标，例如眉毛、眼睛、鼻子、嘴巴和脸部轮廓等位置坐标等。

## 赛事任务

给定人脸图像，找到4个人脸关键点，赛题任务可以视为一个关键点检测问题。

- 训练集：5千张人脸图像，并且给定了具体的人脸关键点标注。
- 测试集：约2千张人脸图像，需要选手识别出具体的关键点位置。


## 赛题数据

赛题数据由训练集和测试集组成，train.csv为训练集标注数据，train.npy和test.npy为训练集图片和测试集图片，可以使用numpy.load进行读取。train.csv的信息为左眼坐标、右眼坐标、鼻子坐标和嘴巴坐标，总共8个点。

本次竞赛的评价标准回归MAE进行评价，数值越小性能更优，最高分为0。评估代码参考：

```
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)
```
