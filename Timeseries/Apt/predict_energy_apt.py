#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:33:46 2020

@author: keti
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
import gc
gc.collect()
import sys, os
os.getcwd()
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')
from pytictoc import TicToc

apt_rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Heat_Analysis/Apt/춘의주공APT.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage', 
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
#                              year=lambda x: x['time'].dt.year.astype('str'),
                              month=lambda x: x['time'].dt.month.astype('str'),
                              hour=lambda x: x['time'].dt.hour.astype('str'),
#                              week= lambda x: x['time'].dt.weekday_name,
                              gas_type=lambda x: x['gas_type'].astype('str'),
#                              raw_usage=lambda x: x['raw_usage'].astype('float64'),
                              calib_usage=lambda x: x['calib_usage'].astype('float'),
                              calib_const=lambda x: x['calib_const'].astype('float'),
                              flux=lambda x: x['flux'].astype('float'),
                              calib_usage_1monthago=lambda x: x['calib_usage'].shift(720),
                              calib_usage_1weekago=lambda x: x['calib_usage'].shift(168),
                              calib_usage_1dayago=lambda x: x['calib_usage'].shift(24))[['time','calib_usage']].dropna()\
                      .query("time>= '2018-12-01 00:00' and time< '2018-12-17 00:00'")\
                      .set_index('time')

apt_rawdata.index.freq= 'H'


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    rolmean = pd.Series(timeseries).rolling(window=24).mean()
    rolstd = pd.Series(timeseries).rolling(window=24).std()
    #Plot rolling statistics:
    fig = plt.figure(figsize=(20, 10))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best'); plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    #Perform Dickey-Fuller test:
    print ('<Results of Dickey-Fuller Test>')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


test_stationarity(apt_rawdata['calib_usage'])
apt_rawdata['first_difference'] = apt_rawdata['calib_usage'] - apt_rawdata['calib_usage'].shift(1)  
test_stationarity(apt_rawdata.first_difference.dropna(inplace=False))
apt_rawdata['seasonal_first_difference'] = apt_rawdata['first_difference'] - apt_rawdata['first_difference'].shift(24)  
test_stationarity(apt_rawdata.seasonal_first_difference.dropna(inplace=False))

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# 정상성 확인 stationarity check
plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(211)
plot_acf(apt_rawdata.calib_usage, lags=40, ax=ax1)
#plot_acf(apt_rawdata.seasonal_first_difference.iloc[25:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
plot_pacf(apt_rawdata.calib_usage,lags=40,ax=ax2)
#plot_pacf(apt_rawdata.seasonal_first_difference,lags=40,ax=ax2)
plt.show()

#Seasonal Decomposition (STL)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(apt_rawdata['calib_usage'], freq=24)  
plt.figure()  
decomposition.plot()

sns.reset_defaults()
plt.clf()



# timeseries modeling========================================
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math
from multiprocessing import cpu_count
from joblib import Parallel,delayed
from warnings import catch_warnings,filterwarnings
from tbats import TBATS, BATS
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def measure_rmse(actual, predicted):
	return math.sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

#gridsearch(ETS)===============================================

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	history = np.array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = exp_smoothing_forecast(history, cfg)# store forecast in list of predictions		
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							models.append(cfg)
	return models

data = apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'].values
n_test = 48
cfg_list = exp_smoothing_configs(seasonal=[0,6,12,24])
scores = grid_search(data, cfg_list, n_test)
for cfg, error in scores[:3]:
	print('Best ExponentialSmoothing%s RMSE=%.3f' % (cfg, error))

# trend변수에 'add'를 설정할 경우 트랜드가 직선으로 증가 혹은 감소, 'mul'를 설정할 경우 트랜드가 지수적으로 증가 혹은 감소
ets_model = ExponentialSmoothing(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'], trend=None, damped=False, seasonal='add', seasonal_periods=24)\
.fit(optimized=True, use_boxcox=False, remove_bias=False)

apt_rawdata['forecast_ets'] = ets_model.predict(start = len(apt_rawdata)-48, end= len(apt_rawdata)-1)  
apt_rawdata.query("time> '2018-12-15 00:00'")[['calib_usage', 'forecast_ets']].plot(figsize=(10, 7))

# grid search(SARIMA)===================================================================================================

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)
 
# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models
 
data = apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'].values
n_test = 48	# data split
cfg_list = sarima_configs(seasonal=[0, 12, 24]) # model configs
scores = grid_search(data, cfg_list, n_test)# grid search
	
# list top 3 configs
for cfg, error in scores[:5]:
	print('Best SARIMA%s MAE=%.3f' % (cfg, error))

sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'],order=(1,0,2), seasonal_order=(2,0,2,24),trend='n').fit()
sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(1,1,2,24),trend='ct').fit()
apt_rawdata['forecast_sarima'] = sarima_model.predict(start = len(apt_rawdata)-48, end= len(apt_rawdata)-1, dynamic= True)  
apt_rawdata.query("time> '2018-12-15 00:00'")[['calib_usage', 'forecast_sarima']].plot(figsize=(10, 7))
apt_rawdata[-48:]
sarima_model.plot_diagnostics(figsize=(10, 7));

#building the model
from pmdarima.arima import auto_arima
arima_model = auto_arima(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'], trace=True, error_action='ignore',seasonal=True, suppress_warnings=True)\
.fit(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'])
apt_rawdata['forecast_arima'] = arima_model.predict(384)
apt_rawdata.query("time> '2018-12-15 00:00'")[['calib_usage', 'forecast_arima']].plot(figsize=(10, 7))
apt_rawdata[-48:]

del(arima_model)

# day and week season
tbats_model = TBATS(seasonal_periods=(1, 24)).fit(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'])
tbats_model = TBATS(seasonal_periods=(24, 168)).fit(apt_rawdata.query("time< '2018-12-15 00:00'")['calib_usage'])
apt_rawdata['forecast_tbats'] = tbats_model.forecast(steps=384)
apt_rawdata.query("time> '2018-12-15 00:00'")[['calib_usage', 'forecast_tbats']].plot(figsize=(10, 7))




apt_rawdata['1day_ago'] = apt_rawdata.shift(24)[-48:]['calib_usage']
apt_rawdata['1week_ago'] = apt_rawdata.shift(168)[-48:]['calib_usage']

apt_rawdata.query("time> '2018-12-15 00:00'")[['calib_usage', '1day_ago']].plot(figsize=(10, 7))
apt_rawdata.query("time> '2018-12-15 00:00'")[['calib_usage', '1week_ago']].plot(figsize=(10, 7))

#validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('RMSE=%.3f' % math.sqrt(mean_squared_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['1day_ago'])))
print('MAE=%.3f' % (mean_absolute_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['1day_ago'])))

print('RMSE=%.3f' % math.sqrt(mean_squared_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['1week_ago'])))
print('MAE=%.3f' % (mean_absolute_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['1week_ago'])))

print('RMSE=%.3f' % math.sqrt(mean_squared_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_ets'])))
print('MAE=%.3f' % (mean_absolute_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_ets'])))

print('RMSE=%.3f' % math.sqrt(mean_squared_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_arima'])))
print('MAE=%.3f' % (mean_absolute_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_arima'])))

print('RMSE=%.3f' % math.sqrt(mean_squared_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_sarima'])))
print('MAE=%.3f' % (mean_absolute_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_sarima'])))

print('RMSE=%.3f' % math.sqrt(mean_squared_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_tbats'])))
print('MAE=%.3f' % (mean_absolute_error(apt_rawdata[-48:]['calib_usage'], apt_rawdata[-48:]['forecast_tbats'])))


# timeseries modeling(end)========================================




# keras Model=====================================================================
## ( multi-layer perceptron 모델)
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
import gc
gc.collect()
import sys, os
os.getcwd()
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')
from pytictoc import TicToc

apt_rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Heat_Analysis/Apt/춘의주공APT_18년1월부터.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage', 
                                        '보정계수(평균)':'calib_const', '순간유량(평균 임시사용)': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time>= '2018-12-01 00:00' and time< '2018-12-31 00:00'").drop('time',axis=1)


def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 72

# 1. 데이터셋 생성하기
signal_data =apt_rawdata

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
signal_data = scaler.fit_transform(signal_data)

# 데이터 분리
train = signal_data[0:360]
val = signal_data[360:504]
test = signal_data[504:]

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

# 2. 모델 구성하기
multiperceptron_model = Sequential()
multiperceptron_model.add(Dense(32,input_dim=72,activation="relu"))
multiperceptron_model.add(Dropout(0.3))

for i in range(2):
    multiperceptron_model.add(Dense(32,activation="relu"))
    multiperceptron_model.add(Dropout(0.3))
multiperceptron_model.add(Dense(1))

# 3. 모델 학습과정 설정하기
multiperceptron_model.compile(loss='mean_squared_error', optimizer='adagrad')

# 4. 모델 학습시키기
multiperceptron_hist = multiperceptron_model.fit(x_train, y_train, epochs=50, batch_size=24, validation_data=(x_val, y_val))
# 5. 학습과정 살펴보기
plt.plot(multiperceptron_hist.history['loss'])
plt.plot(multiperceptron_hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
          
# 6. 모델 평가하기
trainScore = multiperceptron_model.evaluate(x_train, y_train, verbose=0)
print('Train Score: ', trainScore)
valScore = multiperceptron_model.evaluate(x_val, y_val, verbose=0)
print('Validataion Score: ', valScore)
testScore = multiperceptron_model.evaluate(x_test, y_test, verbose=0)
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 48
xhat = x_test[0, None]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = multiperceptron_model.predict(xhat, batch_size=32)
    predictions[i] = prediction
    xhat = np.hstack([xhat[:,1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()

## (순환신경망 모델)
# 0. 사용할 패키지 불러오기
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import gc
gc.collect()
import sys, os
os.getcwd()
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')
from pytictoc import TicToc

apt_rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Heat_Analysis/Apt/춘의주공APT_18년1월부터.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage', 
                                        '보정계수(평균)':'calib_const', '순간유량(평균 임시사용)': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time>= '2018-12-01 00:00' and time< '2018-12-31 00:00'").drop('time',axis=1)

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
apt_rawdata = scaler.fit_transform(apt_rawdata)

look_back = 72

# 데이터 분리
train = apt_rawdata[0:360]
val = apt_rawdata[360:504]
test = apt_rawdata[504:]

def create_dataset(apt_rawdata, look_back=1):
    dataX, dataY = [], []
    for i in range(len(apt_rawdata)-look_back):
        dataX.append(apt_rawdata[i:(i+look_back), 0])
        dataY.append(apt_rawdata[i + look_back, 0])
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
lstm_model = Sequential()
lstm_model.add(LSTM(32, input_shape=(None, 1)))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(1))
# 3. 모델 학습과정 설정하기
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
# 4. 모델 학습시키기
lstm_hist = lstm_model.fit(x_train, y_train, epochs=150, batch_size=24, validation_data=(x_val, y_val))

# 5. 학습과정 살펴보기
plt.plot(lstm_hist.history['loss'])
plt.plot(lstm_hist.history['val_loss'])
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = lstm_model.evaluate(x_train, y_train, verbose=0)
lstm_model.reset_states()
print('Train Score: ', trainScore)
valScore = lstm_model.evaluate(x_val, y_val, verbose=0)
lstm_model.reset_states()
print('Validataion Score: ', valScore)
testScore = lstm_model.evaluate(x_test, y_test, verbose=0)
lstm_model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 48
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = lstm_model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()


## (상태유지 순환신경망 모델)
# 0. 사용할 패키지 불러오기
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
import gc
gc.collect()
import sys, os
os.getcwd()
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')
from pytictoc import TicToc

apt_rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Heat_Analysis/Apt/춘의주공APT_18년1월부터.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage', 
                                        '보정계수(평균)':'calib_const', '순간유량(평균 임시사용)': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time>= '2018-12-01 00:00' and time< '2018-12-31 00:00'").drop('time',axis=1)

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
apt_rawdata = scaler.fit_transform(apt_rawdata)
# 데이터 분리
train = apt_rawdata[0:360]
val = apt_rawdata[360:540]
test = apt_rawdata[540:]


def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
       
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

look_back = 72

# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(100):
    print(i)
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val, y_val))
    model.reset_states()

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
model.reset_states()
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 48
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()


## (상태유지  스택 순환신경망 모델)
# 0. 사용할 패키지 불러오기
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
import gc
gc.collect()
import sys, os
os.getcwd()
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')
from pytictoc import TicToc

apt_rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Heat_Analysis/Apt/춘의주공APT_18년1월부터.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage', 
                                        '보정계수(평균)':'calib_const', '순간유량(평균 임시사용)': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              calib_usage=lambda x: x['calib_usage'].astype('float'))[['time','calib_usage']].dropna()\
                      .query("time>= '2018-12-01 00:00' and time< '2018-12-31 00:00'").drop('time',axis=1)

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
apt_rawdata = scaler.fit_transform(apt_rawdata)
# 데이터 분리
train = apt_rawdata[0:360]
val = apt_rawdata[360:540]
test = apt_rawdata[540:]


def create_dataset(signal_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal_data)-look_back):
        dataX.append(signal_data[i:(i+look_back), 0])
        dataY.append(signal_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

look_back = 48


# 데이터셋 생성
x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 2. 모델 구성하기
model = Sequential()
for i in range(2):
    model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(50):
    print(i)
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist], validation_data=(x_val, y_val))
    model.reset_states()

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
model.reset_states()
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 48
xhat = x_test[0]
predictions = np.zeros((look_ahead,1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:],prediction])
    
plt.figure(figsize=(12,5))
plt.plot(np.arange(look_ahead),predictions,'r',label="prediction")
plt.plot(np.arange(look_ahead),y_test[:look_ahead],label="test function")
plt.legend()
plt.show()






































apt_rawdata_hour = apt_rawdata.groupby('hour').mean().assign(hour=lambda x:x.index)\
.rename(columns={'calib_usage' : 'calib_usage_mean'}).drop(['calib_usage_1dayago','calib_usage_1weekago','calib_usage_1monthago'],axis=1).reset_index(drop=True)

apt_rawdata_hour = apt_rawdata.groupby(['month','hour']).mean().assign(hour=lambda x:x.index)\
.rename(columns={'calib_usage' : 'calib_usage_mean'}).drop(['calib_usage_1dayago','calib_usage_1weekago','calib_usage_1monthago'],axis=1).reset_index(drop=True)

apt_rawdata_hour.info()
apt_rawdata=pd.merge(apt_rawdata, apt_rawdata_hour,on='hour', how='left')


len(apt_rawdata.time.unique())
apt_rawdata.columns
apt_rawdata.head().round(2)
apt_rawdata.describe().round(2)
apt_rawdata.info()
apt_rawdata.isnull().sum()
(apt_rawdata.isnull().sum()/max(apt_rawdata.count())).round(2)

# columns name sort
categorical_feature = list( set([ col for col in apt_rawdata.columns if apt_rawdata[col].dtypes == "object"])-set(['gas_type']))
time_feature = list( set([ col for col in apt_rawdata.columns if apt_rawdata[col].dtypes == "datetime64[ns]"]))
numerical_feature = list(set(apt_rawdata.columns) - set(categorical_feature)-set(time_feature))

minmax_feature = list( set(numerical_feature)-set(['calib_usage']))
target_feature = list(['calib_usage'])
dummy_feature=list(set(categorical_feature))

# Graph
import matplotlib.pyplot as plt
import seaborn as sns

for col in categorical_feature:
    unique_list = apt_rawdata[col].unique()
    print(unique_list)

for col in categorical_feature:
    apt_rawdata[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

for col in numerical_feature:
    sns.distplot(apt_rawdata.loc[apt_rawdata[col].notnull(), col])
    plt.title(col)
    plt.show()

# corr and mic graph
#%matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
apt_rawdata[numerical_feature].plot()
apt_rawdata.drop(['month','hour'],axis=1).plot(subplots=True, fontsize=8) # total
apt_rawdata.query("index>= '2019-01-27 01:00' and index< '2019-02-03 00:00'").drop(['month','hour'],axis=1).plot(subplots=True, fontsize=8) #1week
apt_rawdata.query("index>= '2019-01-27 01:00' and index< '2019-01-28 00:00'").drop(['month','hour'],axis=1).plot(subplots=True, fontsize=8) #1month
apt_rawdata.groupby(apt_rawdata.month).mean().drop(['hour'],axis=1).plot(subplots=True, fontsize=8)
apt_rawdata.groupby(apt_rawdata.hour).mean().drop(['month'],axis=1).plot(subplots=True, fontsize=8)
apt_rawdata.groupby(apt_rawdata.week).mean().drop(['month','hour'],axis=1).plot(subplots=True, fontsize=8)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# 정상성 확인 stationarity check
plot_acf(apt_rawdata.calib_usage)
plot_acf(apt_rawdata.raw_usage)
plot_pacf(apt_rawdata.calib_usage)
plt.show()

#Seasonal Decomposition (STL)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(apt_rawdata['calib_usage'], freq=24)  
fig = plt.figure()  
fig = decomposition.plot()


sns.reset_defaults()
plt.clf()

import matplotlib.pyplot as plt
apt_rawdata["log_usage"] = np.log(apt_rawdata.calib_usage)
apt_rawdata["difflog_usage"] = apt_rawdata.calib_usage.diff()
plt.figure(figsize=(10, 10))
plt.subplot(311)
apt_rawdata.plot(x="time", y='calib_usage')
plt.subplot(312)
apt_rawdata.plot(x="time", y="log_usage")
plt.subplot(313)
apt_rawdata.plot(x="time", y="difflog_usage")
plt.title("houly Apt Gas Usage")
plt.show()

import scipy as sp
x, y = sp.stats.boxcox_normplot(apt_rawdata['calib_usage']+0.00001, -3, 3)
plt.plot(x, y)
y2, l = sp.stats.boxcox(apt_rawdata.calib_usage+0.00001)
plt.axvline(x=l, color='r', ls="--")
plt.show()
print("optimal lambda:", l)

plt.figure(figsize=(20, 10))
sm.graphics.tsa.plot_acf(apt_rawdata['calib_usage'], lags=48, ax=plt.subplot(211))
sm.graphics.tsa.plot_pacf(apt_rawdata['calib_usage'], lags=48, ax=plt.subplot(212),color='g')
plt.xlim(-1, 48)
plt.ylim(-0.5, 1.1)

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)

# Generate all different combinations of p, q and q triplets
import itertools
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

gridsearch=list()

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(apt_rawdata.calib_usage,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            gridsearch.append('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()



ar_model = sm.tsa.ARMA(apt_rawdata.calib_usage, (2, 0)).fit()
ar_model.plot_predict(end="2019-02-20 23:00:00", plot_insample=False)
ar_model.summary()
ma_model = sm.tsa.ARMA(apt_rawdata.calib_usage, (0, 2)).fit()
ma_model.plot_predict(end="2019-02-20 23:00:00", plot_insample=False)
ma_model.summary()
arma_model = sm.tsa.ARMA(apt_rawdata.calib_usage, (2, 1)).fit()
arma_model.plot_predict(end="2019-02-20 23:00:00", plot_insample=False)
arma_model.summary()
arima_model = sm.tsa.ARIMA(apt_rawdata.calib_usage, [2, 1, 1]).fit()
arima_model.plot_predict(end="2019-02-20 23:00:00", plot_insample=False)
arima_model.summary()
sarima_model = sm.tsa.statespace.SARIMAX(apt_rawdata.calib_usage, order=(1, 0, 0), seasonal_order=(1, 1, 1, 24),
                                         enforce_stationarity=False,  enforce_invertibility=False).fit()
sarima_model.summary()
sarima_model.plot_diagnostics()
pred = sarima_model.get_prediction(start=len(apt_rawdata), end=len(apt_rawdata) + 24)


import statsmodels.api as sm
result2 = sm.OLS.from_formula('calib_usage ~ calib_usage_1weekago+ C(month)+ C(hour) - 1', data= apt_rawdata).fit()
print(result2.summary())

result2.predict(X_test)

plt.figure(figsize=(20, 10))
plt.plot(apt_rawdata.calib_usage, label="gas usage 시계열")
plt.plot(result2.fittedvalues, lw=3, alpha=0.5, label="추정한 시계열")
plt.plot(result2.resid, label="잔차")
plt.title("시계열의 계절성과 추세 추정")
plt.legend()
plt.show()

























from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(apt_rawdata[set(apt_rawdata.columns)-set(['calib_usage','time'])],
                                                            apt_rawdata['calib_usage'], test_size=0.3, random_state=777, stratify=apt_rawdata['hour'])
train_y=pd.DataFrame(train_y)
#train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['price'])], rawdata['price'], test_size=0.1, random_state=777, stratify=rawdata['year'])
    
train=pd.concat([train_x, train_y], axis=1) # columns bind
test=pd.concat([test_x, test_y], axis=1) # columns bind

from pytictoc import TicToc
tictoc.tic() #Time elapsed since t.tic()
 # scaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
minmaxscaler = MinMaxScaler().fit(train_x[minmax_feature])
minmaxscaler_target = MinMaxScaler().fit(train_y)
train_minmax = pd.DataFrame(minmaxscaler.transform(train_x[minmax_feature]),
                                columns= minmax_feature, index=list(train_x.index.values))
train_target = pd.DataFrame(minmaxscaler_target.transform(train_y),
                                columns= target_feature, index=list(train_y.index.values))
train_dummy=pd.get_dummies(train_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
train_preprocess=pd.concat([train_minmax, train_dummy], axis=1) # columns bind
train_all=pd.concat([train_preprocess, train_target], axis=1)

#train_x[minmax_feature].describe()
#print(minmaxscaler.data_min_)
#print(minmaxscaler_target.data_min_)
#print(minmaxscaler.data_max_)
#print(minmaxscaler_target.data_max_)

#Correlattion=========================================
corr= train_all.drop((list(set(train_dummy))), axis=1).corr(method='pearson')\
        .loc['calib_usage'].round(3)
       
from minepy import MINE
def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
    print("GMIC", mine.gmic())
    print("TIC", mine.tic())

mine = MINE(alpha=0.6, c=15, est="mic_approx")
mine.compute_score(train_all['calib_usage_1dayago'], train_all['calib_usage'])
print_stats(mine)

# corr and mic graph
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.scatter(train_all['calib_usage_1dayago'], train_all['calib_usage'])
mine.compute_score(train_all['calib_usage_1dayago'], train_all['calib_usage'])
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,0] ))

train_x= train_preprocess
train_y= train_target


test_minmax = pd.DataFrame(minmaxscaler.transform(test_x[minmax_feature]),
                                columns= minmax_feature, index=list(test_x.index.values))
test_target = pd.DataFrame(minmaxscaler_target.transform(pd.DataFrame(test_y)),
                                columns= target_feature, index=list(test_y.index.values))
test_dummy=pd.get_dummies(test_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
test_x=pd.concat([test_minmax, test_dummy], axis=1) # columns bind
test_y=test_target


from sklearn.linear_model import Ridge, Lasso, LinearRegression
#    alpha : float, optional
#         디폴트 1.0,   클수록 가중치 규제 (특성 수 줄이기, 과대적합 방지),   작을수록 정확하게 (과대적합)
#    alpha 0이면 ordinary least square (일반적 리니어리그래션)
#           가중치가 커질수록 특성이 늘어나 훈련 데이터에 과적합됨   alpha 옵션을 크게 할수록 가중치 규제 (가중치의 크기가 못 커지게 하기, 과적합 방지)
#           크기 개념을 l1(맨하튼 거리) 으로   하냐 l2(유클리디안 거리의 제곱)로 생각하냐에 따라  라쏘(l1) 혹은 릿지(l2) 를 사용
linear_model=LinearRegression(fit_intercept=True).fit(train_x, train_y)
lasso_model=Lasso(alpha=0.5).fit(train_x, train_y)
ridge_model =Ridge(alpha=0.5).fit(train_x, train_y)

from sklearn.tree import DecisionTreeRegressor
#    criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'
tree_model= DecisionTreeRegressor().fit(train_x, train_y) 

from sklearn.svm import LinearSVR, SVR, SVC
#    kernel : string, optional (default=’rbf’)
#    linear: 선형 SVM, poly: 다항식 커널 함수 사용, 비선형 SVM, rbf: RBF (Radial Basis Function (방사 기저 함수)) 커널 함수 사용, 비선형 SVM, 디폴트
#    C : float, optional (default=1.0) 클수록 정확하게 (마진이 작아짐, 과대적합)   작을수록 과대적합 방지 (과대적합 방지),  alpha (가중치 규제) 의 역수
#    gamma : float, optional (default=’auto’) 클수록 정확하게 (경사가 급해짐, 과대적합)   작을수록 과대적합 방지 (과대적합 방지)  비선형 (kernel=‘rbf’, kernel=‘poly’)에서만 이 옵션 사용 
#    ex) C=1.0, gamma='auto
linsvc_model= LinearSVR().fit(train_x, train_y) 
#svc_model=SVC(kernel='linear') .fit(train_x, train_y) 
    
from sklearn.ensemble import BaggingRegressor
#    base_estimator: base estimator to fit
#    n_estimators: the number of base estimators 
#    bootstrap : (default=True) 중복 할당 여부  True 베깅, S  False 페이스팅
bagging_model = BaggingRegressor(base_estimator=Ridge(alpha=0.5)).fit(train_x, train_y)

from sklearn.ensemble import RandomForestRegressor
#    n_estimators : integer, optional (default=10) The number of trees in the forest.
#    bootstrap : boolean, optional (default=True) True 베깅,   False 페이스팅
#    criterion : string, optional (default=”mse”) 'mse' (평균제곱오차),   'friedman_mse', 'mae'
rf_model = RandomForestRegressor().fit(train_x, train_y)

from sklearn.ensemble import AdaBoostRegressor
#    base_estimator : object, optional (default=None)
#    n_estimators : integer, optional (default=50) The maximum number of estimators at which boosting is terminated.
adaboost_model = AdaBoostRegressor(base_estimator=Lasso()).fit(train_x, train_y)

from sklearn.ensemble import GradientBoostingRegressor
#    n_estimators : int (default=100) The number of boosting stages to perform. 
#    criterion : string, optional (default=”friedman_mse”) friedman_mse, mse, mae
gradient_model = GradientBoostingRegressor(criterion='mae').fit(train_x, train_y)

import xgboost
xgb_model=xgboost.XGBRegressor(objective ='reg:squarederror').fit(train_x, train_y)  

from lightgbm import LGBMModel,LGBMRegressor
lgb_model=LGBMRegressor().fit(train_x, train_y)

result=test_x.assign(QGEN=test_target,
                     QGEN_linear= lambda x: np.where(linear_model.predict(test_x)<0, 0,
                                                   np.where(linear_model.predict(test_x)>1,1, linear_model.predict(test_x))),
                     QGEN_lasso= lambda x: np.where(lasso_model.predict(test_x)<0, 0,
                                                   np.where(lasso_model.predict(test_x)>1,1, lasso_model.predict(test_x))),                                                     
                     QGEN_ridge= lambda x: np.where(ridge_model.predict(test_x)<0, 0,
                                                   np.where(ridge_model.predict(test_x)>1,1, ridge_model.predict(test_x))),
                     QGEN_linsvc= lambda x: np.where(linsvc_model.predict(test_x)<0, 0,
                                                   np.where(linsvc_model.predict(test_x)>1,1, linsvc_model.predict(test_x))),                                                    
                     QGEN_tree= lambda x: np.where(tree_model.predict(test_x)<0,0,
                                                np.where(tree_model.predict(test_x)>1,1,tree_model.predict(test_x))),
                     QGEN_bagging= lambda x: np.where(bagging_model.predict(test_x)<0,0,
                                                np.where(bagging_model.predict(test_x)>1,1,bagging_model.predict(test_x))),
                     QGEN_adaboost= lambda x: np.where(adaboost_model.predict(test_x)<0,0,
                                                np.where(adaboost_model.predict(test_x)>1,1,adaboost_model.predict(test_x))),                  
                     QGEN_rf= lambda x: np.where(rf_model.predict(test_x)<0,0,
                                                np.where(rf_model.predict(test_x)>1,1,rf_model.predict(test_x))),
                     QGEN_gradient= lambda x: np.where(gradient_model.predict(test_x)<0,0,
                                                np.where(gradient_model.predict(test_x)>1,1,gradient_model.predict(test_x))),                                                 
                     QGEN_xgb= lambda x: np.where(xgb_model.predict(test_x)<0,0,
                                                np.where(xgb_model.predict(test_x)>1,1,xgb_model.predict(test_x))),
                     QGEN_lgb= lambda x: np.where(lgb_model.predict(test_x)<0,0,
                                                np.where(lgb_model.predict(test_x)>1,1,lgb_model.predict(test_x))),
                     QGEN_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN'])+minmaxscaler_target.data_min_,
                     QGEN_1dayago_inverse=lambda x: minmaxscaler_target.data_max_*(x['calib_usage_1dayago'])+minmaxscaler_target.data_min_,
                     QGEN_1weekago_inverse=lambda x: minmaxscaler_target.data_max_*(x['calib_usage_1weekago'])+minmaxscaler_target.data_min_,
                     QGEN_1monthago_inverse=lambda x: minmaxscaler_target.data_max_*(x['calib_usage_1monthago'])+minmaxscaler_target.data_min_,
                     QGEN_linear_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_linear']) + minmaxscaler_target.data_min_,
                     QGEN_lasso_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_lasso'])+minmaxscaler_target.data_min_,
                     QGEN_ridge_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_ridge'])+minmaxscaler_target.data_min_,
                     QGEN_linsvc_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_linsvc']) + minmaxscaler_target.data_min_,
                     QGEN_tree_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_tree'])+minmaxscaler_target.data_min_,
                     QGEN_bagging_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_bagging'])+minmaxscaler_target.data_min_,
                     QGEN_adaboost_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_adaboost']) + minmaxscaler_target.data_min_,
                     QGEN_rf_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_rf'])+minmaxscaler_target.data_min_,
                     QGEN_gradient_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_gradient'])+minmaxscaler_target.data_min_,
                     QGEN_lgb_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_lgb'])+minmaxscaler_target.data_min_,
                     QGEN_xgb_inverse=lambda x: minmaxscaler_target.data_max_*(x['QGEN_xgb'])+minmaxscaler_target.data_min_).drop(set(test_dummy),axis=1)
                                            

from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

        def precision(data,predict,origin):
            Rsquared = r2_score(data[origin],data[predict]).round(3)
            MAE = mean_absolute_error(data[origin],data[predict]).round(3)
            MSE = mean_squared_error(data[origin],data[predict]).round(3)
            RMSE = np.sqrt(mean_squared_error(data[origin],data[predict])).round(3)
            MSLE = mean_squared_log_error(data[origin],data[predict]).round(3)
            RMLSE = np.sqrt(mean_squared_log_error(data[origin],data[predict])).round(3)
            MAPE = np.mean((abs(data[origin]-data[predict]))/(data[origin]+1))*100
            sMAPE = round(np.mean(200*(abs(data[origin]-data[predict]))/(data[origin]+data[predict])),2)
            print(predict,'[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,'MAPE:',MAPE,'sMAPE:',sMAPE,']')
    

        score_model1=['QGEN_linear','QGEN_lasso','QGEN_ridge','QGEN_linsvc','QGEN_tree','QGEN_bagging','QGEN_adaboost','QGEN_rf','QGEN_gradient','QGEN_lgb','QGEN_xgb',
                      'calib_usage_1dayago','calib_usage_1weekago','calib_usage_1monthago']
        score_model2=['QGEN_linear_inverse','QGEN_lasso_inverse','QGEN_ridge_inverse','QGEN_linsvc_inverse','QGEN_tree_inverse',
                      'QGEN_bagging_inverse','QGEN_adaboost_inverse','QGEN_rf_inverse','QGEN_gradient_inverse','QGEN_lgb_inverse', 'QGEN_xgb_inverse',
                      'QGEN_1dayago_inverse','QGEN_1weekago_inverse','QGEN_1monthago_inverse']

        for i in score_model2:
            precision(result.query('QGEN_inverse>1'),i,'QGEN_inverse')
            precision(result,i,'QGEN_inverse')
        
        for i in score_model1:
            precision(result,i,'QGEN')



















#train / test split
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
splits = TimeSeriesSplit(n_splits=5,max_train_size=None)
from matplotlib import pyplot
pyplot.figure(1)
index = 1

Result=pd.DataFrame(columns=(['usage','AR_usage', 'MA_usage', 'ARMA_usage', 'ARIMA_usage', 'SARIMA_usage']))
X=apt_rawdata.calib_usage.astype(float).values
for train_index, test_index in splits.split(X):
	train = X[train_index]
	test = X[test_index]
	print('Observations: %d' % (len(train) + len(test)))
	print('Training Observations: %d' % (len(train)))
	print('Testing Observations: %d' % (len(test)))
	ar_model = sm.tsa.ARMA(apt_rawdata.calib_usage.astype(float).values, (1, 0)).fit()
	ma_model = sm.tsa.ARMA(apt_rawdata.calib_usage.astype(float).values, (0, 1)).fit()
	arma_model = sm.tsa.ARMA(apt_rawdata.calib_usage.astype(float).values, (1, 1)).fit()
	arima_model = sm.tsa.ARIMA(apt_rawdata.calib_usage.astype(float).values, order=(0,1, 1)).fit()
	sarima_model = sm.tsa.statespace.SARIMAX(apt_rawdata.calib_usage.astype(float).values, 
                                             order=(1, 1, 1),
                                             seasonal_order=(1, 1, 1, 12),
                                             enforce_stationarity=False,
                                             enforce_invertibility=False).fit()
	tmp=pd.DataFrame(columns=(['usage','AR_usage', 'MA_usage', 'ARMA_usage', 'ARIMA_usage', 'SARIMA_usage']))\
	.assign(usage=test,
            AR_usage=ar_model.predict(start=1,end=41),
            MA_usage=ma_model.predict(start=1,end=41),
            ARMA_usage=arma_model.predict(start=1,end=41),
            ARIMA_usage=arima_model.predict(start=1,end=41),
            SARIMA_usage=sarima_model.predict(start=1,end=41))
	Result=pd.concat([Result,tmp], axis=0)
	pyplot.subplot(310 + index)
	pyplot.plot(train)
	pyplot.plot([None for i in train] + [x for x in test])
	index += 1
pyplot.show()

ar_model = sm.tsa.ARMA(apt_rawdata.calib_usage.astype(float).values, (1, 0)).fit()
ma_model = sm.tsa.ARMA(apt_rawdata.calib_usage.astype(float).values, (0, 1)).fit()
arma_model = sm.tsa.ARMA(apt_rawdata.calib_usage.astype(float).values, (1, 1)).fit()
arima_model = sm.tsa.ARIMA(apt_rawdata.calib_usage.astype(float).values, order=(0,1, 1)).fit()
sarima_model = sm.tsa.statespace.SARIMAX(apt_rawdata.calib_usage.astype(float).values, 
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit()

print(ar_model.summary())
print(ma_model.summary())
print(arma_model.summary())
print(arima_model.summary())
print(sarima_model.summary())
sarima_model.plot_diagnostics()

Result=pd.DataFrame(columns=(['usage','AR_usage', 'MA_usage', 'ARMA_usage', 'ARIMA_usage', 'SARIMA_usage']))\
.assign(usage=
        AR_usage=ar_model.predict(start=1,end=24),
        MA_usage=ma_model.predict(start=1,end=24),
        ARMA_usage=arma_model.predict(start=1,end=24),
        ARIMA_usage=arima_model.predict(start=1,end=24),
        SARIMA_usage=sarima_model.predict(start=1,end=24))

ar_model.predict(start=1,end=24)
ma_model.predict(start=1,end=24)
arma_model.predict(start=1,end=24)
arima_model.predict(start=1,end=24)
sarima_model.predict(start=1,end=24)


ar_model.forecast(steps=24)
ar_model.plot_predict(plot_insample=False)
ma_model.plot_predict(plot_insample=False)
arma_model.plot_predict(plot_insample=False)
arima_model.plot_predict(plot_insample=False)
#ar.plot_predict(start='1936-01-01', end='1960-01-01', ax=ax, plot_insample=False);



from fbprophet import Prophet
m = Prophet(yearly_seasonality=True)
m.fit(df)





plt.figure(figsize=(25, 15))
plt.subplot(221)
plt.scatter(apt_rawdata['time'], apt_rawdata['raw_usage'],s=2,c='blue', label='raw')
plt.scatter(apt_rawdata['time'], apt_rawdata['calib_usage'],s=2,c='red', label='calib')
plt.legend(['raw','calib'], fontsize=12, loc='upper right')
plt.subplot(222)
plt.plot(apt_rawdata['time'], apt_rawdata['raw_usage'], linestyle='-',marker='o', linewidth=1, color='b', label='raw') # 'dashed'
plt.plot(apt_rawdata['time'], apt_rawdata['calib_usage'], linestyle='-.',marker='o', linewidth=1, color='r', label='calib') # 'dashed'
plt.legend(['raw','calib'], fontsize=12, loc='upper right')
plt.subplot(223)
plt.scatter(apt_rawdata['hour'], apt_rawdata['raw_usage'],s=2,c='blue', label='raw')
plt.scatter(apt_rawdata['hour'], apt_rawdata['calib_usage'],s=2,c='red', label='calib')
plt.legend(['raw','calib'], fontsize=12, loc='upper right')
plt.subplot(224)
plt.plot(apt_rawdata['hour'], apt_rawdata['raw_usage'], linestyle='-',marker='o', linewidth=1, color='b', label='raw') # 'dashed'
plt.plot(apt_rawdata['hour'], apt_rawdata['calib_usage'], linestyle='-.',marker='o', linewidth=1, color='r', label='calib') # 'dashed'
plt.legend(['raw','calib'], fontsize=12, loc='upper right')




plt.subplot(231)
plt.scatter(apt_rawdata['time'], apt_rawdata['raw_usage'],s=1,c='blue')
plt.scatter(apt_rawdata.index, apt_rawdata['raw_usage'],s=1,c='blue')
mine.compute_score(train_all['torque'], train_all['price'],s=1,c='blue')
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,0] ))
plt.subplot(232)
plt.scatter(train_all['weight'], train_all['price'],s=1,c='blue')
mine.compute_score(train_all['weight'], train_all['price'],s=1,c='blue')
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,1] ))
plt.subplot(233)
plt.scatter(train_all['power'], train_all['price'],s=1,c='blue')
mine.compute_score(train_all['power'], train_all['price'],s=1,c='blue')
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,2] ))
plt.subplot(234)
plt.scatter(train_all['displacement'], train_all['price'],s=1,c='blue')
mine.compute_score(train_all['displacement'], train_all['price'],s=1,c='blue')
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,3] ))
plt.subplot(235)
plt.scatter(train_all['efficiency'], train_all['price'],s=1,c='blue')
mine.compute_score(train_all['efficiency'], train_all['price'],s=1,c='blue')
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,4] ))





