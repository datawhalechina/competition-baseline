# IMDB评论剧透检测
## 竞赛链接
https://god.yanxishe.com/20
## score
74.729
## 操作说明
数据放在data目录下
执行ml.ipynb
## 优化方向
baseline中只利用了review_text信息
#### 文本方向
review_summary，以及IMDB_movie_details.json信息进行挖掘
#### 时序方向
review_date进行挖掘
#### 其他方向
movie_id，user_id，rating进行挖掘
