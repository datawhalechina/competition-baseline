import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

train_df = pd.read_csv('../input/train.csv', header=None, sep=';')
test_df = pd.read_csv('../input/test.csv', header=None, sep=';')

train_df = train_df[train_df[11] != 'quality']
lbl = LabelEncoder().fit(train_df[11])
train_df[11] = lbl.transform(train_df[11])

for a, b in combinations([0,1,2,3,4,7,8,9,10], 2):
    train_df[str(a) + '_' + str(b)] = train_df[a].astype(float) + train_df[b].astype(float)
    train_df[str(a) + '/' + str(b)] = train_df[a].astype(float) / train_df[b].astype(float)
    train_df[str(a) + '*' + str(b)] = train_df[a].astype(float) * train_df[b].astype(float)
    train_df[str(a) + '/log' + str(b)] = train_df[a].astype(float) / np.log1p(train_df[b].astype(float))
    
    test_df[str(a) + '_' + str(b)] = test_df[a].astype(float) + test_df[b].astype(float)
    test_df[str(a) + '/' + str(b)] = test_df[a].astype(float) / test_df[b].astype(float)
    test_df[str(a) + '*' + str(b)] = test_df[a].astype(float) * test_df[b].astype(float)
    test_df[str(a) + '/log' + str(b)] = test_df[a].astype(float) / np.log1p(test_df[b].astype(float))
    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

n_fold = 10
skf = StratifiedKFold(n_splits = n_fold, shuffle = True)
eval_fun = roc_auc_score

def run_oof(clf, X_train, y_train, X_test, kf):
    print(clf)
    preds_train = np.zeros((len(X_train), 7), dtype = np.float)
    preds_test = np.zeros((len(X_test), 7), dtype = np.float)
    train_loss = []; test_loss = []

    i = 1
    for train_index, test_index in kf.split(X_train, y_train):
        x_tr = X_train[train_index]; x_te = X_train[test_index]
        y_tr = y_train[train_index]; y_te = y_train[test_index]
        clf.fit(x_tr, y_tr, eval_set = [(x_te, y_te)], early_stopping_rounds = 500, verbose = False)
        
        # train_loss.append(eval_fun(y_tr, clf.predict_proba(x_tr)[:]))
        # test_loss.append(eval_fun(y_te, clf.predict_proba(x_te)[:]))

        preds_train[test_index] = clf.predict_proba(x_te)[:]
        preds_test += clf.predict_proba(X_test)[:]

        # print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(i, train_loss[-1], test_loss[-1], np.mean(test_loss)))
        print('-' * 50)
        i += 1
    print('Train: ', train_loss)
    print('Val: ', test_loss)
    print('-' * 50)
    # print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(np.mean(train_loss), np.mean(test_loss)))
    preds_test /= n_fold
    return preds_train, preds_test

params = {
    'learning_rate': 0.01,
    'min_child_samples': 5,
    'max_depth': 5,
    'lambda_l1': 2,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 3000,
    'metric': 'multi_error',
    'num_class': 7,
    'feature_fraction': .75,
    'bagging_fraction': .85,
    'seed': 99,
    'num_threads': 20,
    'verbose': -1
}

train_pred, test_pred = run_oof(lgb.LGBMClassifier(**params), 
                                train_df.drop(11, axis=1).values, 
                                train_df[11].values, 
                                test_df.values, 
                                skf)

submit = pd.DataFrame()
submit[0] = range(len(test_df))
submit[1] = lbl.inverse_transform(np.argmax(test_pred, 1))
submit.to_csv('lgb.csv', index=None, header=None)
