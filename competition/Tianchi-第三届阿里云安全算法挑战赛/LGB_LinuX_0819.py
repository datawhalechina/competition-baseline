# -*- coding: utf-8 -
"""
Created on Fri Jul 27 19:11:05 2018

@author: liuxiang02
"""
import numpy as np
import pandas as pd
import gc, scipy
import gensim
import os
from gensim.models.word2vec import Word2Vec, Text8Corpus
import os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import KFold
from gensim.models.word2vec import Word2Vec, Text8Corpus
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from tqdm import tqdm, tqdm_notebook
# LSA Feature
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy import sparse

start0 = time.time()
# 只用了前1.5亿条数据,
train = pd.read_csv('./train.csv',encoding='utf-8')#,nrows=2000000
df_train = train.groupby(by='file_id').apply(lambda x:' '.join(x.api))
labels = train.groupby(by='file_id').apply(lambda x:np.unique(x.label)[0]).values
df_train = pd.DataFrame(df_train)
df_train.rename(columns={0:'text'},inplace=True)
print(train.shape, df_train.shape)



test = pd.read_csv('./test.csv',encoding='utf-8')
df_test = test.groupby(by='file_id').apply(lambda x:' '.join(x.api))
df_test = pd.DataFrame(df_test)
df_test.rename(columns={0:'text'},inplace=True)
testIDs = df_test.index
print(test.shape, df_test.shape)
gc.collect()
print(time.time()-start0)

trAPI = df_train['text']
teAPI = df_test['text']

train['api_return'] = train.return_value.map(str)
api_train = train.groupby(by='file_id').apply(lambda x:' '.join(x.api_return))
api_train = pd.DataFrame(api_train)
api_train.rename(columns={0:'return'},inplace=True)

test['api_return'] = test.return_value.map(str)
api_test = test.groupby(by='file_id').apply(lambda x:' '.join(x.api_return))
api_test = pd.DataFrame(api_test)
api_test.rename(columns={0:'return'},inplace=True)

import networkx as nx
apiSet = list(set(train.api) | set(test.api))

# TFIDF特征
print('tfidf starts')
tfidf = False
if tfidf:
    vec = TFIDF(ngram_range=(1, 4), max_features=300000)
    tfidf_train = vec.fit_transform(df_train['text'])
    tfidf_test = vec.transform(df_test['text'])
    print(tfidf_train.shape, tfidf_test.shape, time.time() - start0)

    sparse.save_npz('./virus_set/tfidf_train.npz', tfidf_train)  # 保存
    sparse.save_npz('./virus_set/tfidf_test.npz', tfidf_test)  # 保存

    # TFIDF特征
    # vec = TFIDF(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
    # tfidf_train_api_return = vec.fit_transform(df_train['parallel_api'])
    # tfidf_test_api_return = vec.transform(df_test['parallel_api'])
    # print(tfidf_train_api_return.shape, tfidf_test_api_return.shape,time.time()-start0)

    # sparse.save_npz('./virus_set/tfidf_train_api_return.npz', tfidf_train_api_return)  #保存
    # sparse.save_npz('./virus_set/tfidf_test_api_return.npz', tfidf_test_api_return)    #保存

else:
    tfidf_train = sparse.load_npz('./virus_set/tfidf_train.npz')  # 读
    tfidf_test = sparse.load_npz('./virus_set/tfidf_test.npz')  # 读
    # tfidf_train_api_return = sparse.load_npz('./virus_set/tfidf_train_api_return.npz') #读
    # tfidf_test_api_return = sparse.load_npz('./virus_set/tfidf_test_api_return.npz') #读
print(time.time() - start0)
print(tfidf_train.shape, tfidf_test.shape)


lsa = False
if lsa:
    word_vectorizer = TFIDF(ngram_range=(4,4),max_features=500000)
    #word_vectorizer.fit(df_train['text'].append(df_test['text']))
    X_train_tfidf = word_vectorizer.fit_transform(df_train['text'])
    X_test_tfidf = word_vectorizer.transform(df_test['text'])
    svd = TruncatedSVD(300)
    lsa = make_pipeline(svd,Normalizer(copy=False))

    lsa_train = lsa.fit_transform(X_train_tfidf)
    lsa_test = lsa.fit_transform(X_test_tfidf)

    np.save('./virus_set/lsa_train.npy', lsa_train)  # 保存
    np.save('./virus_set/lsa_test.npy', lsa_test)    # 保存

else:
    lsa_train = np.load('./virus_set/lsa_train.npy')  # 读
    lsa_test = np.load('./virus_set/lsa_test.npy')    # 读
print('lsa result:',lsa_train.shape,lsa_test.shape,time.time()-start0)


value_lsa = False
if value_lsa:
    word_vectorizer = TFIDF(ngram_range=(1,4), min_df=3, max_df=0.9,max_features=300000)
    #word_vectorizer.fit(df_train['text'].append(df_test['text']))
    X_train_tfidf = word_vectorizer.fit_transform(api_train['return'])
    X_test_tfidf = word_vectorizer.transform(api_test['return'])
    svd = TruncatedSVD(500)
    lsa = make_pipeline(svd,Normalizer(copy=False))

    value_lsa_train = lsa.fit_transform(X_train_tfidf)
    value_lsa_test = lsa.fit_transform(X_test_tfidf)


    np.save('./virus_set/value_lsa_train.npy', value_lsa_train)  # 保存
    np.save('./virus_set/value_lsa_test.npy', value_lsa_test)    # 保存

else:
    value_lsa_train = np.load('./virus_set/value_lsa_train.npy')  # 读
    value_lsa_test = np.load('./virus_set/value_lsa_test.npy')    # 读
print('lsa2 result:',value_lsa_train.shape,value_lsa_test.shape,time.time()-start0)

apiValues = dict(train.groupby(train.api)['return_value'].apply(lambda x: len(set(x))))

def chooseFeature0812(df, dfOrigin, part):
    indexs, files = df.index, dfOrigin.file_id.unique()

    for col in apiSet:
        df[col + '_returns'] = 0
        df[col + '_tids'] = 0
        df[col + '_index'] = 0
        df[col + '_returns_ratio'] = 0
    print(df.shape)

    for index, file in zip(indexs, files):
        X = pd.read_csv('./' + part + '/' + str(index) + '.csv')
        X['api_return'] = X.api + X['return_value'].map(str)
        X['api_index'] = X.api + X['index'].map(str)
        X['api_tid'] = X.api + X['tid'].map(str)

        df.loc[index, 'api_types'] = len(set(X['api']))
        df.loc[index, 'api_types_ratio'] = df.loc[index, 'api_types'] / X.shape[0]

        df.loc[index, 'api_return_type'] = len(set(X['api_return']))
        df.loc[index, 'api_return_type_ratio'] = df.loc[index, 'api_return_type'] / X.shape[0]

        df.loc[index, 'api_index_type'] = len(set(X['api_index']))
        df.loc[index, 'api_index_type_ratio'] = df.loc[index, 'api_index_type'] / X.shape[0]

        df.loc[index, 'api_tid_type'] = len(set(X['api_tid']))
        df.loc[index, 'api_tid_type_ratio'] = df.loc[index, 'api_tid_type'] / X.shape[0]

        df.loc[index, 'index_cnts_unique'] = len(set(X['index']))
        df.loc[index, 'index_cnts_unique_bool'] = 1 if df.loc[index, 'index_cnts_unique'] == X.shape[0] else -1
        df.loc[index, 'index_repeat_ratio'] = df.loc[index, 'index_cnts_unique'] / X.shape[0]

        df.loc[index, 'index_repeat'] = X.groupby(X.tid)['index'].apply(lambda x: \
                                                                            1 if len(x) == len(set(x)) else -1).min()
        sameIndexApi = X.groupby(X['index'])['api'].apply(lambda x: len(set(x)))
        df.loc[index, 'sameIndexApiMax'] = sameIndexApi.max()
        df.loc[index, 'sameIndexApiMean'] = sameIndexApi.mean()

        api_return = dict(X.groupby(X.api)['return_value'].apply(lambda x: len(set(x))))
        api_tid = dict(X.groupby(X.api)['tid'].apply(lambda x: len(set(x))))
        api_index = dict(X.groupby(X.api)['index'].apply(lambda x: len(set(x))))
        # print( df.loc[index,'api_types'],len(api_return),len(api_tid),len(api_index))

        for api in api_return:
            df.loc[index, api + '_returns'] = api_return.get(api,0)
            df.loc[index, api + '_tids'] = api_tid.get(api,0)
            df.loc[index, api + '_index'] = api_index.get(api,0)
            df.loc[index, api + '_returns_ratio'] = api_return[api] / (apiValues.get(api,0)+1)
        #print(index, time.time() - start0)

    return df


def chooseFeature(df, dfOrigin, part):
    indexs, files = df.index, dfOrigin.file_id.unique()
    # Basic features
    colNames = ['return_min', 'return_max', 'return_median', 'return_sum', 'return_cnts', 'return_cnts_unique',
                'return_avg', 'tid_min', 'tid_max', 'tid_median', 'tid_cnts', 'tid_cnts_unique', 'tid_avg']
    for col in colNames: df[col] = 0

    for index, file in zip(indexs, files):
        X = pd.read_csv('./' + part + '/' + str(index) + '.csv')
        X['api_return'] = X.api + X['return_value'].map(str)

        # df.loc[index, 'api_types'] = len(set(X['api']))
        df.loc[index, 'index_max'] = X['index'].max()
        df.loc[index, 'index_min'] = X['index'].min()
        df.loc[index, 'index_cnts_unique'] = len(set(X['index']))

        # new feature
        df.loc[index, 'index_repeat_ratio'] = df.loc[index, 'index_cnts_unique'] / X.shape[0]

        df.loc[index, 'return_min'] = X.return_value.min()
        df.loc[index, 'return_max'] = X.return_value.max()
        df.loc[index, 'return_median'] = X.return_value.median()
        df.loc[index, 'return_std'] = X.return_value.std()
        df.loc[index, 'return_sum'] = X.return_value.sum()
        df.loc[index, 'return_cnts'] = len(X.return_value)
        df.loc[index, 'return_cnts_unique'] = len(set(X.return_value))
        df.loc[index, 'return_avg'] = len(X.return_value) / (len(np.unique(X.return_value)) + 1)

        df.loc[index, 'return_gt_0'] = np.sum(X.return_value > 0)
        df.loc[index, 'return_eq_0'] = np.sum(X.return_value == 0)
        df.loc[index, 'return_lt_0'] = np.sum(X.return_value < 0)
        df.loc[index, 'return_lt_0_ratio'] = df.loc[index, 'return_gt_0'] / X.shape[0]
        df.loc[index, 'return_eq_0_ratio'] = df.loc[index, 'return_eq_0'] / X.shape[0]

        df.loc[index, 'return_eq_1'] = np.sum(X.return_value == 1)
        df.loc[index, 'return_eq_2'] = np.sum(X.return_value == 2)
        df.loc[index, 'return_eq_1_ratio'] = df.loc[index, 'return_eq_1'] / X.shape[0]
        df.loc[index, 'return_eq_2_ratio'] = df.loc[index, 'return_eq_2'] / X.shape[0]

        df.loc[index, 'tid_min'] = X.tid.min()
        df.loc[index, 'tid_max'] = X.tid.max()
        df.loc[index, 'tid_median'] = X.tid.median()
        df.loc[index, 'tid_cnts'] = len(X.tid)
        df.loc[index, 'tid_cnts_unique'] = len(np.unique(X.tid))
        df.loc[index, 'tid_avg'] = df.loc[index, 'tid_cnts'] / (df.loc[index, 'tid_cnts_unique'] + 1)

        C = X.groupby(X.tid).apply(lambda x: len(x))
        Y = X.groupby(X.tid)['api'].apply(lambda x: len(x.unique()))
        S = X.groupby(X.tid)['return_value'].apply(lambda x: len(x.unique()))
        O = X.groupby(X.tid)['api_return'].apply(lambda x: len(x.unique()))

        df.loc[index, 'tid_max_apis'], df.loc[index, 'tid_min_apis'], df.loc[index, 'tid_avg_apis'] = np.max(C), np.min(
            C), np.mean(C)
        df.loc[index, 'api_tid_min'], df.loc[index, 'api_tid_max'], df.loc[index, 'api_tid_mean'] = np.mean(Y), np.max(
            Y), np.min(Y)
        df.loc[index, 'return_tid_min'], df.loc[index, 'api_tid_max'], df.loc[index, 'api_tid_mean'] = np.mean(
            S), np.max(S), np.min(S)
        df.loc[index, 'api_return_tid_min'], df.loc[index, 'api_tid_max'], df.loc[index, 'api_tid_mean'] = np.mean(
            O), np.max(O), np.min(O)

        # groupby by same index in different tids
        # M = pd.DataFrame(X.groupby(by='index').apply(lambda x:' '.join(x.api)))
        # M.rename(columns={0:'api_paral'},inplace=True)
        # df.loc[index,'parallel_api'] = ' '.join(M.api_paral)

        print(index, time.time() - start0)
    return df

basic = False
if basic:
    df_train = chooseFeature0812(df_train, train, 'train')
    df_test = chooseFeature0812(df_test, test, 'test')

    print('222222222222222')
    df_train = chooseFeature(df_train, train, 'train')
    df_test = chooseFeature(df_test, test, 'test')
    print(df_train.shape, df_test.shape, time.time() - start0)
    df_train.drop('text',axis=1).to_csv('./virus_set/df_train.csv',index=None,encoding='utf-8')
    df_test.drop('text',axis=1).to_csv('./virus_set/df_test.csv',index=None,encoding='utf-8')
else:

    df_train = pd.read_csv('./virus_set/df_train.csv', encoding='utf-8')
    df_test = pd.read_csv('./virus_set/df_test.csv', encoding='utf-8')
print('basic:',df_train.shape, df_test.shape)

df_train['text']= trAPI
df_test['text'] = teAPI

print('222222222222222,graph start')
from networkx.algorithms import bipartite
def DirectedGraphFeature(df, dfOrigin, part):
    indexs, files = df.index, dfOrigin.file_id.unique()
    for col in apiSet:
        df[col + '_graph_degree'] = 0
        df[col + '_graph_in_degree'] = 0
        df[col + '_graph_cental_degree'] = 0
        df[col + '_graph_center'] = 0

    for index, file in zip(indexs, files):
        X = pd.read_csv('./' + part + '/' + str(index) + '.csv')
        api = X.groupby(by='tid').apply(lambda x: ' '.join(x.api))
        api = pd.DataFrame(api)
        api.rename(columns={0: 'api_call'}, inplace=True)

        G = nx.DiGraph()
        for row in api.index:
            apiCall = (api.loc[row, 'api_call']).split(' ')
            for i in range(len(apiCall) - 1):
                G.add_edge(apiCall[i], apiCall[i + 1])

        if (len(G) <= 1):
            continue

        df.loc[index, 'dir_density'] = nx.density(G)
        df.loc[index, 'dir_order'] = G.order()
        df.loc[index, 'dir_size'] = G.size()

        #(left, right, cover_size) = {}, {}, 0
        #if (bipartite.is_bipartite(G)):
            #left, right = nx.bipartite.sets(G)
            #matching = nx.bipartite.maximum_matching(G)
            #vertex_cover = nx.bipartite.to_vertex_cover(G, matching)
            #cover_size = len(set(G) - vertex_cover)
            #print(left, right, cover_size)

        #df.loc[index, 'dir_left'] = len(left)
        #df.loc[index, 'dir_gright'] = len(right)
        #df.loc[index, 'dir_graph_degree'] = cover_size

        df.loc[index, 'dir_dominating_set'] = len(set(nx.dominating_set(G)))
        df.loc[index, 'dir_transitivity'] = nx.transitivity(G)
        df.loc[index, 'dir_is_strongly_connected'] = int(nx.is_strongly_connected(G))
        df.loc[index, 'dir_is_weakly_connected'] = int(nx.is_weakly_connected(G))

        Degree = G.degree()
        OutDegree = nx.out_degree_centrality(G)
        CentalDrgree = nx.degree_centrality(G)

        for x in G.nodes():
            df.loc[index, x+'_graph_degree'] = Degree[x]
            df.loc[index, x+'_graph_in_degree'] = OutDegree[x]
            df.loc[index, x+'_graph_cental_degree'] = CentalDrgree[x]

        print(index)
    return df


def UndirectedGraphFeature(df, dfOrigin, part):
    indexs, files = df.index, dfOrigin.file_id.unique()
    for col in apiSet:
        df[col+'_center_degree'] = 0

    for index, file in zip(indexs, files):
        X = pd.read_csv('./' + part + '/' + str(index) + '.csv')
        api = X.groupby(by='tid').apply(lambda x: ' '.join(x.api))
        api = pd.DataFrame(api)
        api.rename(columns={0: 'api_call'}, inplace=True)

        G = nx.Graph()
        for row in api.index:
            apiCall = (api.loc[row, 'api_call']).split(' ')
            for i in range(len(apiCall) - 1):
                G.add_edge(apiCall[i], apiCall[i + 1])
        if (len(G) <= 1):
            continue
        isConnnected = (nx.is_connected(G) == False)
        if isConnnected:
            df.loc[index, 'avg_length'] = -1
            df.loc[index, 'minimum_edge_cut'] = -1
            df.loc[index, 'degree_assortativity_coefficient'] = -1
            df.loc[index, 'radius'] = -1
            df.loc[index, 'diameter'] = -1
            df.loc[index, 'periphery'] = -1
            df.loc[index, 'is_eulerian'] = -1
            df.loc[index, 'center'] = -1
            df.loc[index, 'order'] = -1
            df.loc[index, 'size'] = -1
            df.loc[index, 'density'] = -1

        else:
            df.loc[index, 'avg_length'] = nx.average_shortest_path_length(G)
            df.loc[index, 'minimum_edge_cut'] =  len(set(nx.minimum_edge_cut(G)))
            df.loc[index, 'degree_assortativity_coefficient'] = nx.degree_assortativity_coefficient(G)
            df.loc[index, 'radius'] = nx.radius(G)
            df.loc[index, 'diameter'] = nx.diameter(G)
            df.loc[index, 'periphery'] = len(set(nx.periphery(G)))
            df.loc[index, 'is_eulerian'] = int(nx.is_eulerian(G))
            df.loc[index, 'center'] =  len(set(nx.center(G)))
            df.loc[index, 'density'] = nx.density(G)
            df.loc[index, 'order'] = G.order()
            df.loc[index, 'size'] = G.size()

        if not isConnnected:
            for x in set(nx.center(G)):
                df.loc[index,x+'_center_degree'] = 1
        print(index)
    return df

graph = False
if graph:
    df_train = UndirectedGraphFeature(df_train,train,'train')
    print(time.time()-start0,df_train.shape)
    df_train = DirectedGraphFeature(df_train,train,'train')
    print(time.time()-start0,df_train.shape)
    df_train.fillna(-999, inplace=True)
    df_train.drop('text',axis=1).to_csv('./virus_set/graph_train.csv',index=None,encoding='utf-8')


    df_test = UndirectedGraphFeature(df_test,test,'test')
    print(time.time()-start0,df_test.shape)
    df_test = DirectedGraphFeature(df_test,test,'test')
    print(time.time()-start0,df_test.shape)
    df_test.fillna(-999, inplace=True)
    df_test.drop('text',axis=1).to_csv('./virus_set/graph_test.csv',index=None,encoding='utf-8')
else:
     print('discarding graph')
print('No graph:',df_train.shape, df_test.shape)

#df_train['text']= trAPI
#df_test['text'] = teAPI



def seekLocalFiles(df, dfOrigin, part):
    indexs, files = df.index, dfOrigin.file_id.unique()
    for index, file in zip(indexs, files):
        X = pd.read_csv('./' + part + '/' + str(index) + '.csv')
        apiDic.append(dict(X.api.value_counts()))
        apiSetDic.append(set(X.api))

        returnDic.append(dict(X.return_value.value_counts()))
        returnSetDic.append(set(X.return_value))

    return df


def lengthOfLongestSubstring(s):
    res = 0
    if s is None or len(s) == 0:
        return res
    d = {}
    tmp = 0
    start = 0
    for i in range(len(s)):
        if s[i] in d and d[s[i]] >= start:
            start = d[s[i]] + 1
        tmp = i - start + 1
        d[s[i]] = i
        res = max(res, tmp)
    return res


def maxRepeatLength(arr, a):
    arr = np.array(arr)
    start = []
    for i in range(len(arr)):
        if (i == 0 and arr[i] == a):
            start.append(i)
        elif arr[i] == a and arr[i - 1] != a:
            start.append(i)
        elif i < len(arr) - 1 and arr[i + 1] != a and arr[i] == a:
            start.append(i)
        elif i == len(arr) - 1 and arr[i] == a:
            start.append(i)
    arrLen = len(start)
    if (arrLen <= 1):
        return arrLen
    else:
        cnts = 1
        for i in range(arrLen - 1):
            if (start[i + 1] - start[i] < cnts):
                continue
            S = list(arr[start[i]:start[i + 1]])
            if (S.count(a) == start[i + 1] - start[i]):
                cnts = max(cnts, start[i + 1] - start[i] + 1)
    return cnts


begin = time.time()


def chooseFeature0819(df, dfOrigin, part, apiDic, apiSetDic, returnDic, returnSetDic):
    indexs, files = df.index, dfOrigin.file_id.unique()
    for apiCall in apiSet:
        df[apiCall+'repeatLength'] = 0

    for index, file in zip(indexs, files):
        X = pd.read_csv('./' + part + '/' + str(index) + '.csv')

        api = dict(X.api.value_counts())
        apiTypes = set(X.api)
        returnVal = dict(X.return_value.value_counts())
        returnTypeVal = set(X.return_value)
        if part == 'train':
            df.loc[index, 'apiSameCnts'] = apiDic.count(api) - 1
            df.loc[index, 'apiTypeSameCnts'] = apiSetDic.count(apiTypes) - 1
            df.loc[index, 'returnSameCnts'] = returnDic.count(returnVal) - 1
            df.loc[index, 'returnTypeSameCnts'] = returnSetDic.count(returnTypeVal) - 1
        else:
            df.loc[index, 'apiSameCnts'] = apiDic.count(api)
            df.loc[index, 'apiTypeSameCnts'] = apiSetDic.count(apiTypes)
            df.loc[index, 'returnSameCnts'] = returnDic.count(returnVal)
            df.loc[index, 'returnTypeSameCnts'] = returnSetDic.count(returnTypeVal)

        maxRepeats = 0
        for api in set(X.api):
            df.loc[index,api+'repeatLength'] = maxRepeatLength(X.api.values, api)
            maxRepeats = max(maxRepeats, df.loc[index, api+'repeatLength'])

        df.loc[index, 'maxRepeats'] = maxRepeats
        df.loc[index, 'LLS'] = lengthOfLongestSubstring(X.api.values)
        df.loc[index, 'maxRepeats_LLS_Ratio'] = maxRepeats / df.loc[index, 'LLS']
    return df


substring = True
if substring:

    apiDic, apiSetDic, returnDic, returnSetDic = [], [], [], []
    df_train = seekLocalFiles(df_train, train, 'train')

    df_train = chooseFeature0819(df_train, train, 'train', apiDic, apiSetDic, returnDic, returnSetDic)
    df_train.fillna(0, inplace=True)
    df_train.drop('text',axis=1).to_csv('./virus_set/lls_train.csv',index=None,encoding='utf-8')
    print('lls train: ',time.time()-start0)

    df_test = chooseFeature0819(df_test, test, 'test', apiDic, apiSetDic, returnDic, returnSetDic)
    df_test.fillna(0, inplace=True)
    df_test.drop('text',axis=1).to_csv('./virus_set/lls_test.csv',index=None,encoding='utf-8')
    print('lls test: ',time.time()-start0)

else:
    df_train = pd.read_csv('./virus_set/lls_train.csv', encoding='utf-8')
    df_test = pd.read_csv('./virus_set/lls_test.csv', encoding='utf-8')
print('add lls:',df_train.shape, df_test.shape)



print('222222222222222,lda start')
import numpy as np
import pandas as pd
import gc
import gensim
import time
import lightgbm as lgb
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from scipy.sparse import hstack


# LDA特征
def preprocess(comment):
    return gensim.utils.simple_preprocess(comment, deacc=True, min_len=1)

apiLda = False
if apiLda:
    train_text = df_train['text'].copy().apply(lambda x: preprocess(x))
    test_text = df_test['text'].copy().apply(lambda x: preprocess(x))
    All_text = train_text.append(test_text)
    dictionary = Dictionary(All_text)
    print("There are", len(dictionary), "number of words in the final dictionary")
    corpus = [dictionary.doc2bow(text) for text in All_text]
    ldamodel = LdaModel(corpus=corpus, num_topics=100, id2word=dictionary)

    # creating the topic probability matrix
    topic_probability_mat = ldamodel[corpus]
    train_matrix = topic_probability_mat[:df_train.shape[0]]
    test_matrix = topic_probability_mat[df_train.shape[0]:]
    lda_train = gensim.matutils.corpus2csc(train_matrix).T
    lda_test = gensim.matutils.corpus2csc(test_matrix).T

    sparse.save_npz('./virus_set/lda_train.npz', lda_train)  #保存
    sparse.save_npz('./virus_set/lda_test.npz', lda_test)    #保存
else:
    lda_train = sparse.load_npz('./virus_set/lda_train.npz') #读
    lda_test = sparse.load_npz('./virus_set/lda_test.npz') #读





# drop text,train,test
#df_train.drop(['text'], axis=1, inplace=True)
#df_test.drop(['text'], axis=1, inplace=True)
del train, test
gc.collect()

# 合并所有特征
train_x = hstack([df_train,tfidf_train,lda_train], format='csr')  # ,bow_train,lda_train
test_x = hstack([df_test,tfidf_test,lda_test], format='csr')  # ,bow_test,lda_test
print(train_x.shape, test_x.shape)


params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 6,
    'metric': {'multi_logloss'},
    'learning_rate': 0.02,
    'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 10,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
    #'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
}

# x_train, x_valid, y_train, y_valid = train_test_split(train_x,labels, test_size=0.25, random_state=2018)

# 使用交叉验证
num_folds = 6
predict = np.zeros((test_x.shape[0], 6))
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
preRes = []
for train_index, test_index in skf.split(train_x, labels):
    kfold_y_train, kfold_y_test = labels[train_index], labels[test_index]
    kfold_X_train = train_x[train_index]
    kfold_X_valid = train_x[test_index]
    lgb_train = lgb.Dataset(kfold_X_train, label=kfold_y_train)
    lgb_eval = lgb.Dataset(kfold_X_valid, label=kfold_y_test)
    model = lgb.train(params, train_set=lgb_train, num_boost_round=2000,
                      valid_sets=[lgb_eval], valid_names=['eval'],
                      verbose_eval=20, early_stopping_rounds=30)
    preRes.append(model.best_score)
    predict += model.predict(test_x,model.best_iteration) / num_folds

for loss in preRes:
    print(loss)
resCols = ['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5']
res = pd.DataFrame(predict, columns=['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5'])
res['file_id'] = testIDs
res[resCols].to_csv('./lgb_linux_0820.csv', index=None, encoding='utf-8')
