# TinyMind人民币面值&冠字号编码识别挑战赛

https://www.tinymind.cn/competitions/47

任务1面值分类100分代码，和任务2编码识别第五名代码。

- 任务1：直接是一个分类问题；
- 任务2：可以抽象成一个字符识别问题；
  - 先用检测模型（Fast-RCNN）进行检测；
  - 再使用识别模型CRNN或者muti-CNN进行识别
