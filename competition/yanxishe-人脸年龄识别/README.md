比赛链接：https://god.yanxishe.com/10

使用IMDB-WIKI数据集进行pretrain，再到比赛数据集finetune；
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

单模型4折，就可以达到25准确率，复现Top5成绩；

```
python3 1_train.py
python3 2_predict.py
```

人脸年龄识别练习赛冠军源码_1575964312087.zip为比赛前三名的代码；
