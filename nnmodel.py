import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Dense,Conv2D,Flatten
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

path = 'data/'
train_data = pd.read_csv(path+'nn_data_train.csv',sep=' ')
test_data = pd.read_csv(path + 'nn_data_test.csv',sep = ' ')

print(train_data.head())
print(test_data.head())

labels = ['happiness','id']
cols = train_data.columns
features = [col for col in cols if col not in labels]

y_train = np.array(train_data['happiness'])
x_train = np.array(train_data[features])
x_test = np.array(test_data[features])

def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(0.0001)))

    model.compile(loss='mse', optimizer=Adam(), metrics=['mse'])
    return model

kfolder = KFold(n_splits= 10 ,  random_state=2020,shuffle=True)
oof_nn = np.zeros(len(x_train))
prediction_nn =np.zeros(len(x_test))
kfold = kfolder.split(x_train,y_train)
fold=0


for train_index,val_index in kfold:
    k_x_train = x_train[train_index]
    k_y_train = y_train[train_index]
    k_x_val = x_train[val_index]
    k_y_val = y_train[val_index]

    model = build_model()
    model.fit(k_x_train,k_y_train,validation_data=(k_x_val,k_y_val),batch_size=512,epochs= 2000)
    oof_nn[val_index] = model.predict(k_x_val).reshape((model.predict(k_x_val).shape[0],))
    prediction_nn+=model.predict(x_test).reshape((model.predict(x_test).shape[0],))/kfolder.n_splits

print("scores:{:<8.8f}".format(mean_squared_error(oof_nn,y_train)))

#验证集输出
sub = pd.DataFrame()
sub['id'] = train_data.id
sub['happiness'] = oof_nn
sub.to_csv(path+'oofresult.csv',index=False)

#测试集输出
sub = pd.DataFrame()
sub['id'] = test_data.id
sub['happiness'] = prediction_nn
sub.to_csv(path+'result.csv',index=False)


