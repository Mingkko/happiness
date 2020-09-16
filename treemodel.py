import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold,RepeatedKFold
from tensorflow.keras.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

path = 'data/'
train_data = pd.read_csv(path+'tree_data_train.csv',sep=' ')
test_data = pd.read_csv(path + 'tree_data_test.csv',sep = ' ')

labels = ['happiness','id']
cols = train_data.columns
features = [col for col in cols if col not in labels]

y_train = np.array(train_data['happiness'])
x_train = np.array(train_data[features])
x_test = np.array(test_data[features])

print(train_data.head())
print(test_data.head())


# xgb

xgb_params = {'eta': 0.05, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

folds = KFold(n_splits=10, shuffle=True, random_state=2020)
kfold = folds.split(x_train,y_train)
oof_xgb = np.zeros(len(x_train))
predictions_xgb = np.zeros(len(x_test))

for trn_idx, val_idx in kfold:
    trn_data = xgb.DMatrix(x_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(x_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(x_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(x_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("xgb score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

#lgb
param = {
'num_leaves': 80,
'min_data_in_leaf': 40,
'objective':'regression',
'max_depth': -1,
'learning_rate': 0.1,
"min_child_samples": 30,
"boosting": "gbdt",
"feature_fraction": 0.9,
"bagging_freq": 2,
"bagging_fraction": 0.9,
"bagging_seed": 2029,
"metric": 'mse',
"lambda_l1": 0.1,
"lambda_l2": 0.2,
"verbosity": -1}

folds = KFold(n_splits=10, shuffle=True, random_state=2020)
kfold = folds.split(x_train,y_train)
oof_lgb = np.zeros(len(x_train))
predictions_lgb = np.zeros(len(x_test))

for trn_idx, val_idx in kfold:
    trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 200)
    oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits

print("lgb score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train)))

#stacking
train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
test_stack = np.vstack([predictions_xgb,predictions_lgb]).transpose()

folds_stack = RepeatedKFold(n_splits=5,n_repeats=2,random_state=2020)
oof_stack = np.zeros(train_stack.shape[0])
predictions_stack = np.zeros(len(test_stack))
kfolds = folds_stack.split(train_stack,y_train)

for trn_idx, val_idx in kfolds:
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = BayesianRidge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions_stack += clf_3.predict(test_stack) / 10

print("stack scores:{:8.8f}".format(mean_squared_error(oof_stack, y_train)))

#验证集输出
sub = pd.DataFrame()
sub['id'] = train_data.id
sub['happiness'] = oof_stack
sub.to_csv(path+'treeoofresult.csv',index=False)

#测试集输出
sub = pd.DataFrame()
sub['id'] = test_data.id
sub['happiness'] = predictions_stack
sub.to_csv(path+'treeresult.csv',index=False)
