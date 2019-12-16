思路：

参考https://github.com/ProHiryu/bert-chinese-ner

训练和预测可以修改如下参数：

```
flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training."
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False,"Whether to run the model in inference mode on the test set.")
```

具体用法https://github.com/ProHiryu/bert-chinese-ner
