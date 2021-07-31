### Part1 讯飞赛事介绍

2021年A.I.开发者大赛继续秉承 “技术顶天、应用立地”的坚定理念，开放科大讯飞海量数据资源及人工智能核心技术，全面升级A.I.算法赛、A.I.应用赛、A.I.公益赛、A.I.缤纷赛四大赛道，面向全球开发者，激发人工智能多个行业领域应用的创新探索与挑战。

算法赛与应用赛全方位覆盖了智能语音、CV、NLP、OCR、AR人机交互等人工智能热门研究，同时深耕农业养殖、生物与环保、医疗健康、地理遥感、企业数字化、新能源汽车、金融信息化、智慧城市等多领域多行业方向，期待开发者们尽情展示算法与应用的智慧演练！
除了面向全球专业开发者的数据算法及创新应用两大经典赛道，为进一步赋能行业与生活场景，2021 iFLYTEK A.I.开发者大赛针对赛道进行了创新性升级，丰富的赛题内容带给选手更多的可能性，下面就让我们看看今年都有哪些赛题吧！


### Part2 学术论文分类挑战赛

- 赛题类型：自然语言处理
- 赛题任务：文本分类
- 赛题链接：http://challenge.xfyun.cn/topic/info?type=academic-paper-classification&ch=dw-sq-1

#### 赛题背景 & 任务

随着人工智能技术不断发展，每周都有非常多的论文公开发布。现如今对论文进行分类逐渐成为非常现实的问题，这也是研究人员和研究机构每天都面临的问题。现在希望选手能构建一个论文分类模型。

本次赛题希望参赛选手利用论文信息：论文id、标题、摘要，划分论文具体类别。

```
paperid：9821
title：Calculation of prompt diphoton production cross sections at Tevatron and LHC energies
abstract：A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy.
categories：hep-ph
```

训练数据和测试集以csv文件给出，其中：
- 训练集5W篇论文。其中每篇论文都包含论文id、标题、摘要和类别四个字段。
- 测试集1W篇论文。其中每篇论文都包含论文id、标题、摘要，不包含论文类别字段。

本次竞赛的评价标准采用准确率指标，最高分为1。计算方法参考https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html， 评估代码参考：

```
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
```

#### 赛题解题思路

赛题是一个典型的文本分类任务，所以可以使用文本分类的思路来完成。在本赛题我们主要尝试两个思路：
- 方法1：文本TFIDF特征抽取 + 线性分类
- 方法2：Bert模型文本分类

### Part3 中文问题相似度挑战赛

http://challenge.xfyun.cn/topic/info?type=chinese-question-similarity&ch=dw-sq-1

#### 赛事背景 & 任务

问答系统中包括三个主要的部分：问题理解，信息检索和答案抽取。而问题理解是问答系统的第一部分也是非常关键的一部分。问题理解有非常广泛的应用，如重复评论识别、相似问题识别等。

重复问题检测是一个常见的文本挖掘任务，在很多实际问答社区都有相应的应用。重复问题检测可以方便进行问题的答案聚合，以及问题答案推荐，自动QA等。由于中文词语的多样性和灵活性，本赛题需要选手构建一个重复问题识别算法。

本次赛题希望参赛选手对两个问题完成相似度打分。

- 训练集：约5千条问题对和标签。若两个问题是相同的问题，标签为1；否则为0。
- 测试集：约5千条问题对，需要选手预测标签。

#### 赛题解题思路

赛题是一个典型的文本匹配任务，可以使用文本匹配的思路完成。在本赛题我们主要尝试两个思路：
- 方法1：相似度特征 + 树模型分类
- 方法2：Bert NSP任务

##### 方法1：相似度特征 + 树模型分类

- 文本长度特征
```python
data['q1_len']=data['q1'].astype(str).map(len)
data['q2_len']=data['q2'].astype(str).map(len)
```

- 长度差特征：差/比例
```python
data['q1q2_len_diff']=data['q1_len']-data['q2_len']
data['q1q2_len_diff_abs']=np.abs(data['q1_len']-data['q2_len'])
data['q1q2_rate']=data['q1_len']/data['q2_len']
data['q2q1_rate']=data['q2_len']/data['q1_len']
```

- 特殊符号特征

```python
data['q1_end_special']=data['q1'].str.endswith('？').astype(int)
data['q2_end_special']=data['q2'].str.endswith('？').astype(int)
```

- 共现字特征

```python
data['comm_q1q2char_nums'] = data.apply(lambda  row:len(set(row['q1'])&set(row['q2'])),axis=1)

def char_match_pos(q1, q2, pos_i):
    q1 = list(q1)
    q2 = list(q2)
    if pos_i < len(q1):
        q2_len = min(len(q2), 25)  # q2_len只匹配前25个字
        for pos_j in range(q2_len):
            if q1[pos_i] == q2[pos_j]:
                q_pos = pos_j + 1  # 如果匹配上了 记录匹配的位置
                break
            elif pos_j == q2_len - 1:
                q_pos = 0  # 如果没有匹配上 赋值为0
    else:
        q_pos = -1  # 如果后续长度不存在 赋值为-1

    return q_pos

for pos_i in range(8):
    data['q1_pos_' + str(pos_i + 1)] = data.apply(
        lambda row: char_match_pos(row['q1'], row['q2'], pos_i), axis=1).astype(np.int8)
```

- 距离特征

```python
sim_func_dict = {"jaccard": distance.jaccard,
                 "sorensen": distance.sorensen,
                 "levenshtein": distance.levenshtein,
                 "ratio": Levenshtein.ratio}
for sim_func in tqdm(sim_func_dict, desc="距离特征"):
    data[sim_func] = data.apply(lambda row: sim_func_dict[sim_func](row["q1"],row["q2"]), axis=1)
    qt = [[3, 3], [3, 5], [5, 5], [5, 10], [10, 10], [10, 15], [15, 15], [15, 25]]
    for qt_len in qt:
        if qt_len[0] == 3 and sim_func == "levenshtein":
            pass
        else:
            data[sim_func + '_q' + str(qt_len[0]) + '_t' + str(qt_len[1])] = data.apply(
                lambda row: sim_func_dict[sim_func](row["q1"][:qt_len[0]],
                                                    row["q2"][:qt_len[1]]),
                axis=1)
```

- 向量特征

```
def w2v_sent2vec(words):
    """计算句子的平均word2vec向量, sentences是一个句子, 句向量最后会归一化"""
    M = []
    for word in words:
        try:
            M.append(w2v_model.wv[word])
        except KeyError:  # 不在词典里
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return (v / np.sqrt((v ** 2).sum())).astype(np.float32).tolist()
fea_names = ['q1_vec_{}'.format(i) for i in range(100)]
data[fea_names] = data.progress_apply(lambda row: w2v_sent2vec(row['q1_words_list']), result_type='expand', axis=1)
fea_names = ['q2_vec_{}'.format(i) for i in range(100)]
data[fea_names] = data.progress_apply(lambda row: w2v_sent2vec(row['q2_words_list']), result_type='expand', axis=1)
```

提取完特征后将特征送入LightGBM完成训练即可，线上得分可以有0.84+。

##### 方法2：Bert NSP任务

赛题为经典的文本匹配任务，所以可以考虑使用Bert的NSP来完成建模。

- 步骤1：读取数据集

```
import pandas as pd
import codecs
train_df = pd.read_csv('train.csv', sep='\t', names=['question1', 'question2', 'label'])
```

并按照标签划分验证集：

```
# stratify 按照标签进行采样，训练集和验证部分同分布
q1_train, q1_val, q2_train, q2_val, train_label, test_label =  train_test_split(
    train_df['question1'].iloc[:], 
    train_df['question2'].iloc[:],
    train_df['label'].iloc[:],
    test_size=0.1, 
    stratify=train_df['label'].iloc[:])
```

- 步骤2：文本进行tokenizer

使用Bert对文本进行转换，此时模型选择`bert-base-chinese`。

```
# pip install transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer(list(q1_train), list(q2_train), 
                           truncation=True, padding=True, max_length=100)
val_encoding = tokenizer(list(q1_val), list(q2_val), 
                          truncation=True, padding=True, max_length=100)
```

- 步骤3：定义dataset

```
# 数据集读取
class XFeiDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = XFeiDataset(train_encoding, list(train_label))
val_dataset = XFeiDataset(val_encoding, list(test_label))
```

- 步骤4：定义匹配模型

使用`BertForNextSentencePrediction`完成文本匹配任务，并定义优化器。

```
from transformers import BertForNextSentencePrediction, AdamW, get_linear_schedule_with_warmup
model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 单个读取到批量读取
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# 优化方法
optim = AdamW(model.parameters(), lr=1e-5)
```

- 步骤5：模型训练与验证

祖传代码：模型正向传播和准确率计算。

```
# 训练函数
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        
        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 参数更新
        optim.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))
    
def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in val_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f"%(total_eval_loss/len(val_dataloader)))
    print("-------------------------------")
    

for epoch in range(5):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()
    torch.save(model.state_dict(), f'model_{epoch}.pt')
```

- 步骤6：对测试集进行预测

读取测试集数据，进行转换。

```
test_df = pd.read_csv('test.csv', sep='\t', names=['question1', 'question2', 'label'])
test_df['label'] = test_df['label'].fillna(0)

test_encoding = tokenizer(list(test_df['question1']), list(test_df['question2']), 
                          truncation=True, padding=True, max_length=100)
test_dataset = XFeiDataset(test_encoding, list(test_df['label']))
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
```

对测试集数据进行正向传播预测，得到预测结果，并输出指定格式。

```
def predict():
    model.eval()
    test_predict = []
    for batch in test_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        test_predict += list(np.argmax(logits, axis=1).flatten())
    return test_predict
    
test_label = predict()
pd.DataFrame({'label':test_label}).to_csv('submit.csv', index=None)
```