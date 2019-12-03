import os, sys, codecs
import lightgbm as lgb

def modelWarpper(clf, data_train, data_label, basescore):
    params = {
        'learning_rate': 0.01,
        'min_child_samples': 5,
        'max_depth': 4,
        'lambda_l1': 2,
        'boosting': 'gbdt',
        'objective': 'binary',
        'n_estimators': 2000,
        'metric': 'auc',
        # 'num_class': 6,
        'feature_fraction': .85,
        'bagging_fraction': .85,
        'seed': 99,
        'num_threads': -1,
        'verbose': -1
    }
    for col in data_train.columns:
        cv_results1 = lgb.cv(
                params,
                lgb.Dataset(data_train.drop([col], axis=1).values, label=data_label.values),
                num_boost_round=2000,
                nfold=7, verbose_eval=False,
                early_stopping_rounds=200,
        )
        
        if cv_results1['auc-mean'][-1] > basescore:
            print('+', col, 'CV AUC: ', len(cv_results1['auc-mean']), cv_results1['auc-mean'][-1])
        else:
            print('-', col, 'CV AUC: ', len(cv_results1['auc-mean']), cv_results1['auc-mean'][-1])
            
    XX