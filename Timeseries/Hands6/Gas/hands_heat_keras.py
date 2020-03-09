#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## ( multi-layer perceptron 모델)
import gc
gc.collect()
import sys, os
os.getcwd()
import warnings
warnings.filterwarnings('ignore') # 경고 출력하지 않음

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from pytictoc import TicToc # bench mark

np.random.seed(1295)
# 1. 데이터셋 생성하기
rawdata = pd.DataFrame(pd.read_csv('hands6/heat/핸즈식스2공장_3번.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              year=lambda x: x['time'].dt.year.astype('str'),
                              month=lambda x: x['time'].dt.month.astype('str'),
                              hour=lambda x: x['time'].dt.hour.astype('str'),
                              week= lambda x: x['time'].dt.weekday_name,
                              calib_usage=lambda x: x['calib_usage'].str.replace(',','').astype('float'))\
                              .query("time> '2017-01-01 00:00' and time<= '2020-01-29 00:00'")[['calib_usage','time']]
   # replace
len(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(rawdata.index))
diff = pd.DataFrame( columns=['time'] )
diff['time'] = list(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(rawdata.time))
diff['calib_usage']=None
rawdata=pd.concat([rawdata, diff],axis=0)

rawdata=rawdata.sort_values(by='time')

rawdata.isnull().sum()


rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)
rawdata['calib_usage'] = np.where(pd.notnull(rawdata['calib_usage']) == True, rawdata['calib_usage'], rawdata['calib_usage_1dayago'])
rawdata['calib_usage'] = np.where(pd.notnull(rawdata['calib_usage']) == True, rawdata['calib_usage'], rawdata['calib_usage_1weekago'])
rawdata= rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1)
rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)
rawdata['calib_usage'] = np.where(rawdata['calib_usage'] <500, rawdata['calib_usage'], rawdata['calib_usage_1weekago'])
rawdata['calib_usage'] = np.where(rawdata['calib_usage'] >0, rawdata['calib_usage'], rawdata['calib_usage_1weekago'])
rawdata= rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1)
rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)
rawdata['calib_usage'] = np.where(rawdata['calib_usage'] >0, rawdata['calib_usage'], rawdata['calib_usage_1weekago'])
rawdata= rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1)



rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)
#rawdata['calib_usage_1weekafter']= rawdata['calib_usage'].shift(-168)
#rawdata['calib_usage_1dayafter']= rawdata['calib_usage'].shift(-24)

rawdata['calib_usage'] = np.where(rawdata['time'].dt.date.isin(holiday['date'])==False, rawdata['calib_usage'], 
       np.where(pd.notnull(rawdata['calib_usage_1weekago'])==True,rawdata['calib_usage_1weekago'],rawdata['calib_usage']))
#from datetime import datetime, timedelta
#rawdata['calib_usage'] = np.where((rawdata['time'].dt.date.isin(holiday['date']-timedelta(days=-1))==False), 
#       rawdata['calib_usage'], rawdata['calib_usage_1weekago'])
#rawdata['calib_usage'] = np.where((rawdata['time'].dt.date.isin(holiday['date']-timedelta(days=1))==False), 
#       rawdata['calib_usage'], rawdata['calib_usage_1weekago'])

rawdata =rawdata.set_index('time')
rawdata= rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1)


data=rawdata
data['calib_usage_1dayago']= data['calib_usage'].shift(24)
data['calib_usage_1weekago']= data['calib_usage'].shift(168)
data.plot(figsize=(20, 10))


# 2017.05.18~ 2017.08.10
data1 = data.query("time> '2017-01-09 00:00' and time<= '2017-05-08 00:00'")[['calib_usage']]
data1.plot(figsize=(20, 10))
predict1 = data.query("time> '2017-01-02 00:00' and time<= '2017-01-09 00:00'")


#predict1[['calib_usage', 'calib_usage_1dayago']].plot(figsize=(20, 10))
predict1[['calib_usage', 'calib_usage_1weekago']].plot(figsize=(20, 10))

del(diff, holiday)
gc.collect()


def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


look_back = 504# input 변수로 넣을 시점
look_ahead=168# 예측 시점
# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(data1)
# 데이터 분리( per a day)
train = signal_data[0:len(data1.query("time <= '2017-04-10 00:00'"))] # rawdata.index[0:round(len(signal_data)*0.6)]
val = signal_data[len(data1.query("time <= '2017-03-27 00:00'")):len(data1.query("time <= '2017-04-24 00:00'"))] # rawdata.index[round(len(signal_data)*0.6):round(len(signal_data)*0.95)]
test = signal_data[len(data1.query("time <= '2017-01-09 00:00'")):len(data1.query("time <= '2017-02-13 00:00'"))] # rawdata.index[round(len(signal_data)*0.955):]
#test = signal_data[-(look_back+look_ahead):]

pd.DataFrame(test).plot()
# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

multiperceptron_model = Sequential()
multiperceptron_model.add(Dense(32,input_dim=look_back,activation="relu"))
multiperceptron_model.add(Dropout(0.3))

for i in range(2):
    multiperceptron_model.add(Dense(32,activation="relu"))
    multiperceptron_model.add(Dropout(0.3))
multiperceptron_model.add(Dense(1))


# 3. 모델 학습과정 설정하기
from keras import losses, metrics
#metrics=['mae', 'acc']
#adagrad=keras.optimizers.Adagrad()
#multiperceptron_model.compile(loss='mean_absolute_error', optimizer=adagrad)
#multiperceptron_model.compile(loss='mean_absolute_error', optimizer='Adadelta')
rms=keras.optimizers.RMSprop()
multiperceptron_model.compile(loss='mean_squared_error', optimizer=rms,metrics=['mae','mape'])
# SGD(Stochastic Gradient Descent) Learning rate, Momentum :local minima에 빠지지 않기위해 이전 단계에서의 가중치가 적용된 평균을 사용,  Nesterov Momentum : solution에 가까워 질 수록 gradient를 slow down시킴
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# ADAM(Adaptive Moment Estimation)이전 step에서의 평균뿐 아니라 분산까지 고려한 복잡한 지수 감쇠(exponential decay)를 사용
# adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# RMSProp(Root Mean Squeared Error) 말그대로 지수 감쇠 squared gradients의 평균으로 나눔으로써 learning rate를 감소시킴
# rms=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# Adagrad는 모델 파라미터별 학습률을 사용하는 옵티마이저로, 파라미터의 값이 업데이트되는 빈도에 의해 학습률이 결정. 파라미터가 더 자주 업데이트될수록, 더 작은 학습률이 사용.
# adagrad=keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# Adadelta 과거의 모든 그래디언트를 축적하는 대신, 그래디언트 업데이트의 이동창(moving window)에 기반하여 학습률을 조절.
# adadelta=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

# 4. 모델 학습시키기
multiperceptron_hist = multiperceptron_model.fit(x_train, y_train, epochs=50, batch_size=24, validation_data=(x_val, y_val))
# 5. 학습과정 살펴보기
plt.plot(multiperceptron_hist.history['loss'])
plt.plot(multiperceptron_hist.history['val_loss'])
plt.ylim(0.0, 0.1)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
          
# 6. 모델 평가하기
trainScore = multiperceptron_model.evaluate(x_train, y_train, verbose=0)
print('Train Score: (MSE:%.3f, MAE:%.3f, MAPE:%.3f)' %(trainScore[0],trainScore[1],trainScore[2]))
valScore = multiperceptron_model.evaluate(x_val, y_val, verbose=0)
print('Validataion Score: (MSE:%.3f, MAE:%.3f, MAPE:%.3f)' %(valScore[0],valScore[1],trainScore[2]))
testScore = multiperceptron_model.evaluate(x_test, y_test, verbose=0)
print('Test Score: (MSE:%.3f, MAE:%.3f, MAPE:%.3f)' %(trainScore[0],trainScore[1],trainScore[2]))

# 7. 모델 사용하기
xhat = x_test[0, None]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = multiperceptron_model.predict(xhat, batch_size=24)
    predictions[i] = prediction
    xhat = np.hstack([xhat[:,1:],prediction])

predictions=np.where(predictions<0,0,predictions)
predictions= scaler.data_max_*(predictions)+scaler.data_min_
y_test[:look_ahead]= scaler.data_max_*(y_test[:look_ahead])+scaler.data_min_

plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()

result = data1.query("time > '2017-02-06 00:00' and time <= '2017-02-13 00:00'")
result['forecast_MLP']=predictions
result['time']=result.index
#result['weekday']=result['time'].dt.weekday_name
#result['forecast_MLP']=np.where(result['weekday'].isin( ['Sunday', 'Saturday'])==True,20, result['forecast_MLP'])
result=result.reset_index(drop=True)
result[['calib_usage','forecast_MLP']].plot()
result[['calib_usage','calib_usage_1weekago']].plot()

from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
def precision(predict,origin):
    Rsquared = r2_score(origin,predict).round(3)
    MAE = mean_absolute_error(origin,predict).round(3)
    MSE = mean_squared_error(origin,predict).round(3)
    RMSE = np.sqrt(mean_squared_error(origin,predict)).round(3)
    MSLE = mean_squared_log_error(origin,predict).round(3)
    RMLSE = np.sqrt(mean_squared_log_error(origin,predict)).round(3)
    MAPE = round(np.mean((abs(origin-predict))/(origin))*100,2)
    MAPE_adjust = round(np.mean((abs(origin-predict))/(origin+1))*100,2)
    sMAPE = round(np.mean(200*(abs(origin-predict))/(origin+ predict)),2)
    
    print('[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,'MAPE:',MAPE,'MAPE_adjust:',MAPE_adjust,'sMAPE:',sMAPE,']')

precision(result['forecast_MLP'],result['calib_usage'])

import pymongo
connection=pymongo.MongoClient('10.0.3.36', 27017)
db= connection.get_database('Demand')
for i in pd.date_range(start=result.time[0], end=result.time[len(result.time)-1], freq='H'):
    print(i)
    k=result.query("time=='"+str(i)+"'").forecast_MLP
db.hands6_demand23_predict.update({"time":result[['time']]},{'$push':{'forecast_MLP':result[['forecast_MLP']]}}, True)

pd.to_datetime(result.time)
result[['time']].to_dict('records')
result.time.to_dict('records')
a=result.to_dict('records').keys()

result=result.query("time=='2017-05-07 01:00:00'")

pd.date_range(start ='2018-01-25 01:00:00', end = '2020-01-16 00:00:00', freq = 'H')
 

## (순환신경망 모델)
import gc
gc.collect()
import sys, os
os.getcwd()
import warnings
warnings.filterwarnings('ignore') # 경고 출력하지 않음

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from pytictoc import TicToc # bench mark

# 1. 데이터셋 생성하기
rawdata = pd.DataFrame( pd.read_csv('Apt/춘의주공APT.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
#                              month=lambda x: x['time'].dt.month,
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time> '2018-01-25 00:00' and time<= '2020-01-16 00:00'")
#                      .query("month in [11,12,1,2]").drop(['time','month'],axis=1)
#                      .query("time>= '2018-11-01 00:00' and time< '2019-02-01 00:00'").drop('time',axis=1)





# replace
len(set(pd.date_range(start ='2018-01-25 01:00:00', end = '2020-01-16 00:00:00', freq = 'H'))-set(rawdata.index))
diff = pd.DataFrame( columns=['time'] )
diff['time'] = list(set(pd.date_range(start ='2018-01-25 01:00:00', end = '2020-01-16 00:00:00', freq = 'H'))-set(rawdata.time))
diff['calib_usage']=None
rawdata=pd.concat([rawdata, diff],axis=0)

rawdata=rawdata.sort_values(by='time')

rawdata.isnull().sum()
rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)
rawdata['calib_usage'] = np.where(pd.notnull( rawdata['calib_usage']) == True, rawdata['calib_usage'], rawdata['calib_usage_1dayago'])
rawdata['calib_usage'] = np.where(pd.notnull( rawdata['calib_usage']) == True, rawdata['calib_usage'], rawdata['calib_usage_1weekago'])

rawdata=rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1).set_index('time')

del(diff)
gc.collect()
# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(rawdata)

look_back = 96

# 데이터 분리( per a day)
train = signal_data[0:len(rawdata.query("time <= '2019-12-15 00:00'"))] # rawdata.index[0:round(len(signal_data)*0.6)]
val = signal_data[len(rawdata.query("time > '2019-12-15 00:00'")):len(rawdata.query("time <= '2020-01-01 00:00'"))] # rawdata.index[round(len(signal_data)*0.6):round(len(signal_data)*0.95)]
test = signal_data[len(rawdata.query("time > '2019-12-15 00:00'")):] # rawdata.index[round(len(signal_data)*0.955):]

#signal_data[-274:]

def create_dataset(rawdata, look_back=1):
    dataX, dataY = [], []
    for i in range(len(rawdata)-look_back):
        dataX.append(rawdata[i:(i+look_back), 0])
        dataY.append(rawdata[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)
# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# 2. 모델 구성하기
lstm_model1 = Sequential()
lstm_model1.add(LSTM(32, input_shape=(None, 1)))
lstm_model1.add(Dropout(0.3))
lstm_model1.add(Dense(1))
# 3. 모델 학습과정 설정하기
lstm_model1.compile(loss='mean_squared_error', optimizer='adam')
# 4. 모델 학습시키기
lstm_hist1 = lstm_model1.fit(x_train, y_train, epochs=50, batch_size=12, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
plt.plot(lstm_hist1.history['loss'])
plt.plot(lstm_hist1.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = lstm_model1.evaluate(x_train, y_train, verbose=0)
lstm_model1.reset_states()
print('Train Score: ', round(trainScore,2))
valScore = lstm_model1.evaluate(x_val, y_val, verbose=0)
lstm_model1.reset_states()
print('Validataion Score: ', round(valScore,2))
testScore = lstm_model1.evaluate(x_test, y_test, verbose=0)
lstm_model1.reset_states()
print('Test Score: ', round(testScore,2))

# 7. 모델 사용하기
look_ahead=48
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = lstm_model1.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])

predictions=np.where(predictions<0,0,predictions)
predictions= scaler.data_max_*(predictions)+scaler.data_min_
y_test[:look_ahead]= scaler.data_max_*(y_test[:look_ahead])+scaler.data_min_

plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()


from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
def precision(predict,origin):
    Rsquared = r2_score(origin,predict).round(3)
    MAE = mean_absolute_error(origin,predict).round(3)
    MSE = mean_squared_error(origin,predict).round(3)
    RMSE = np.sqrt(mean_squared_error(origin,predict)).round(3)
    MSLE = mean_squared_log_error(origin,predict).round(3)
    RMLSE = np.sqrt(mean_squared_log_error(origin,predict)).round(3)
    MAPE = round(np.mean((abs(origin-predict))/(origin))*100,2)
    MAPE_adjust = round(np.mean((abs(origin-predict))/(origin+1))*100,2)
    sMAPE = round(np.mean(200*(abs(origin-predict))/(origin+ predict)),2)
    
    print('[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,'MAPE:',MAPE,'MAPE_adjust:',MAPE_adjust,'sMAPE:',sMAPE,']')

precision(predictions,y_test[:look_ahead])


## (상태유지 순환신경망 모델)
import gc
gc.collect()
import sys, os
os.getcwd()
import warnings
warnings.filterwarnings('ignore') # 경고 출력하지 않음

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from pytictoc import TicToc # bench mark

# 1. 데이터셋 생성하기
rawdata = pd.DataFrame( pd.read_csv('Apt/춘의주공APT.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
#                              month=lambda x: x['time'].dt.month,
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time> '2018-01-25 00:00' and time<= '2020-01-16 00:00'")
#                      .query("month in [11,12,1,2]").drop(['time','month'],axis=1)
#                      .query("time>= '2018-11-01 00:00' and time< '2019-02-01 00:00'").drop('time',axis=1)


# replace
len(set(pd.date_range(start ='2018-01-25 01:00:00', end = '2020-01-16 00:00:00', freq = 'H'))-set(rawdata.index))
diff = pd.DataFrame( columns=['time'] )
diff['time'] = list(set(pd.date_range(start ='2018-01-25 01:00:00', end = '2020-01-16 00:00:00', freq = 'H'))-set(rawdata.time))
diff['calib_usage']=None
rawdata=pd.concat([rawdata, diff],axis=0)

rawdata=rawdata.sort_values(by='time')

rawdata.isnull().sum()
rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)
rawdata['calib_usage'] = np.where(pd.notnull( rawdata['calib_usage']) == True, rawdata['calib_usage'], rawdata['calib_usage_1dayago'])
rawdata['calib_usage'] = np.where(pd.notnull( rawdata['calib_usage']) == True, rawdata['calib_usage'], rawdata['calib_usage_1weekago'])

rawdata=rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1).set_index('time')

del(diff)
gc.collect()
# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(rawdata)

look_back = 96

# 데이터 분리( per a day)
train = signal_data[0:len(rawdata.query("time <= '2019-12-15 00:00'"))] # rawdata.index[0:round(len(signal_data)*0.6)]
val = signal_data[len(rawdata.query("time > '2019-12-15 00:00'")):len(rawdata.query("time <= '2020-01-01 00:00'"))] # rawdata.index[round(len(signal_data)*0.6):round(len(signal_data)*0.95)]
test = signal_data[len(rawdata.query("time > '2019-12-15 00:00'")):] # rawdata.index[round(len(signal_data)*0.955):]


def create_dataset(rawdata, look_back=1):
    dataX, dataY = [], []
    for i in range(len(rawdata)-look_back):
        dataX.append(rawdata[i:(i+look_back), 0])
        dataY.append(rawdata[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
       
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)
# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
lstm_model2 = Sequential()
lstm_model2.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
lstm_model2.add(Dropout(0.3))
lstm_model2.add(Dense(1))
# 3. 모델 학습과정 설정하기
lstm_model2.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(50):
    print(i)
    lstm_model2.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val, y_val))
    lstm_model2.reset_states()

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = lstm_model2.evaluate(x_train, y_train, batch_size=1, verbose=0)
lstm_model2.reset_states()
print('Train Score: ', trainScore)
valScore = lstm_model2.evaluate(x_val, y_val, batch_size=1, verbose=0)
lstm_model2.reset_states()
print('Validataion Score: ', valScore)
testScore = lstm_model2.evaluate(x_test, y_test, batch_size=1, verbose=0)
lstm_model2.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 48
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = lstm_model2.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()

from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
def precision(predict,origin):
    Rsquared = r2_score(origin,predict).round(3)
    MAE = mean_absolute_error(origin,predict).round(3)
    MSE = mean_squared_error(origin,predict).round(3)
    RMSE = np.sqrt(mean_squared_error(origin,predict)).round(3)
    MSLE = mean_squared_log_error(origin,predict).round(3)
    RMLSE = np.sqrt(mean_squared_log_error(origin,predict)).round(3)
    MAPE = round(np.mean((abs(origin-predict))/(origin))*100,2)
    MAPE_adjust = round(np.mean((abs(origin-predict))/(origin+1))*100,2)
    sMAPE = round(np.mean(200*(abs(origin-predict))/(origin+ predict)),2)
    
    print('[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,'MAPE:',MAPE,'MAPE_adjust:',MAPE_adjust,'sMAPE:',sMAPE,']')

precision(predictions,y_test[:look_ahead])



## (상태유지 스택 순환신경망 모델)
import gc
gc.collect()
import sys, os
os.getcwd()
import warnings
warnings.filterwarnings('ignore') # 경고 출력하지 않음

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from pytictoc import TicToc # bench mark

# 1. 데이터셋 생성하기
rawdata = pd.DataFrame( pd.read_csv('Apt/춘의주공APT.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
#                              month=lambda x: x['time'].dt.month,
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time>= '2018-11-01 00:00' and time< '2019-02-01 00:00'").drop('time',axis=1)
#                      .query("month in [11,12,1,2]").drop(['time','month'],axis=1)
#                      .query("time>= '2018-11-01 00:00' and time< '2019-02-01 00:00'").drop('time',axis=1)

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(rawdata)

look_back = 168

# 데이터 분리
train = signal_data[0:round(len(signal_data)*0.6)]
val = signal_data[round(len(signal_data)*0.6):round(len(signal_data)*0.8)]
test = signal_data[round(len(signal_data)*0.8):]

def create_dataset(apt_rawdata, look_back=1):
    dataX, dataY = [], []
    for i in range(len(apt_rawdata)-look_back):
        dataX.append(apt_rawdata[i:(i+look_back), 0])
        dataY.append(apt_rawdata[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)
# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
lstm_model3 = Sequential()
for i in range(2):
    lstm_model3.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
    lstm_model3.add(Dropout(0.3))
lstm_model3.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
lstm_model3.add(Dropout(0.3))
lstm_model3.add(Dense(1))
# 3. 모델 학습과정 설정하기
lstm_model3.compile(loss='mean_squared_error', optimizer='adam')
# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(50):
    print(i)
    lstm_model3.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val, y_val))
    lstm_model3.reset_states()

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = lstm_model3.evaluate(x_train, y_train, batch_size=1, verbose=0)
lstm_model3.reset_states()
print('Train Score: ', trainScore)
valScore = lstm_model3.evaluate(x_val, y_val, batch_size=1, verbose=0)
lstm_model3.reset_states()
print('Validataion Score: ', valScore)
testScore = lstm_model3.evaluate(x_test, y_test, batch_size=1, verbose=0)
lstm_model3.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 48
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = lstm_model3.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()




