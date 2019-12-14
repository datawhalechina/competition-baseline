使用IMDB-WIKI数据集进行pretrain，再到比赛数据集finetune；
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

单模型4折，就可以达到25准确率，复现Top5成绩；

```
python3 1_train.py
python3 2_predict.py
```
