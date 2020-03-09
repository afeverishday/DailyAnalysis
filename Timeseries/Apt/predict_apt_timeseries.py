#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:00:29 2020

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

apt_rawdata = pd.DataFrame( pd.read_csv('Hotel/수원라마다호텔(사우나).csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage', 
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              year=lambda x: x['time'].dt.year.astype('str'),
                              month=lambda x: x['time'].dt.month.astype('str'),
                              hour=lambda x: x['time'].dt.hour.astype('str'),
                              week= lambda x: x['time'].dt.weekday_name,
                              raw_usage=lambda x: x['raw_usage'].str.replace(',','').astype('float64'),
                              calib_usage=lambda x: x['calib_usage'].str.replace(',','').astype('float'),
                              calib_const=lambda x: x['calib_const'].astype('float'),
                              flux=lambda x: x['flux'].astype('float'))[['time','calib_usage']]\
                              .query("time>= '2018-01-01 00:00' and time< '2020-01-01 00:00'").set_index('time').dropna()


apt_rawdata = pd.DataFrame(pd.read_csv('hands6/heat/핸즈식스2공장_3번.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              year=lambda x: x['time'].dt.year.astype('str'),
                              month=lambda x: x['time'].dt.month.astype('str'),
                              hour=lambda x: x['time'].dt.hour.astype('str'),
                              week= lambda x: x['time'].dt.weekday_name,
                              calib_usage=lambda x: x['calib_usage'].str.replace(',','').astype('float'))[['time','calib_usage']]\
                              .query("time> '2017-01-01 00:00' and time<= '2020-01-29 00:00'").dropna()

apt_rawdata = pd.DataFrame(pd.read_csv('Apt/춘의주공APT.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                                        '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              year=lambda x: x['time'].dt.year.astype('str'),
                              month=lambda x: x['time'].dt.month.astype('str'),
                              hour=lambda x: x['time'].dt.hour.astype('str'),
                              week= lambda x: x['time'].dt.weekday_name,
                              calib_usage=lambda x: x['calib_usage'].astype('float'))\
                              .query("time> '2018-01-25 00:00' and time<= '2020-01-16 00:00'")\
                              
# replace
len(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(apt_rawdata.index))
diff = pd.DataFrame( columns=['time'] )
diff['time'] = list(set(pd.date_range(start ='2018-01-25 01:00:00', end = '2020-01-16 00:00:00', freq = 'H'))-set(apt_rawdata.time))
diff['calib_usage']=None
apt_rawdata=pd.concat([apt_rawdata, diff],axis=0)

apt_rawdata=apt_rawdata.sort_values(by='time')

apt_rawdata.isnull().sum()
apt_rawdata['calib_usage_1dayago']=apt_rawdata['calib_usage'].shift(24)
apt_rawdata['calib_usage_1weekago']=apt_rawdata['calib_usage'].shift(168)
apt_rawdata['calib_usage'] = np.where(pd.notnull(apt_rawdata['calib_usage']) == True, apt_rawdata['calib_usage'], apt_rawdata['calib_usage_1dayago'])
apt_rawdata['calib_usage'] = np.where(pd.notnull(apt_rawdata['calib_usage']) == True, apt_rawdata['calib_usage'], apt_rawdata['calib_usage_1weekago'])

apt_rawdata=apt_rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1)

del(diff)
gc.collect()
#apt_rawdata.index.freq= 'H'


apt_rawdata =apt_rawdata.query("month in ['11','12','1','2']")[['time','calib_usage']].dropna().set_index('time')

apt_rawdata.shape
apt_rawdata.index
apt_rawdata.columns
apt_rawdata.info()
apt_rawdata.count()
apt_rawdata.describe().round(2).transpose()
apt_rawdata.nunique() 
apt_rawdata.head().round(2)
apt_rawdata.isnull().sum()
(apt_rawdata.isnull().sum()/max(apt_rawdata.count())).round(2)


apt_rawdata.plot.hist()
apt_rawdata.plot.scatter()


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

#import statsmodels.api as sm
#from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing
#fig = plt.figure(figsize=(20,15))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(apt_rawdata.seasonal_first_difference.iloc[25:], lags=40, ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(apt_rawdata.seasonal_first_difference.iloc[25:],lags=40,ax=ax2)


import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# 정상성 확인 stationarity check
fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(211)
plot_acf(apt_rawdata.calib_usage, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
plot_pacf(apt_rawdata.calib_usage,lags=40,ax=ax2)


# lag Plots
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})
fig, axes = plt.subplots(3, 8, figsize=(15,15), sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:24]):
    lag_plot(apt_rawdata.calib_usage, lag=i+1, ax=ax, c='firebrick', alpha=0.5, s=3)
    ax.set_title('Lag ' + str(i+1))

fig.suptitle('Lag Plot', y=1.15)    


from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(apt_rawdata, model='multiplicative')
fig = result.plot()
plot_mpl(fig)


from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

def precision(data,predict,origin):
     Rsquared = r2_score(data[origin],data[predict]).round(2)
     MAE = mean_absolute_error(data[origin],data[predict]).round(2)
     MSE = mean_squared_error(data[origin],data[predict]).round(2)
     RMSE = np.sqrt(mean_squared_error(data[origin],data[predict])).round(2)
     MSLE = mean_squared_log_error(data[origin],data[predict]).round(2)
     RMLSE = np.sqrt(mean_squared_log_error(data[origin],data[predict])).round(2)
     MAPE = round(np.mean((abs(data[origin]-data[predict]))/(data[origin]))*100,2)
     MAPE_adjust = round(np.mean((abs(data[origin]-data[predict]))/(data[origin]+1))*100,2)
     sMAPE = round(np.mean(200*(abs(data[origin]-data[predict]))/(data[origin]+data[predict])),2)
     print(predict,'[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,\
                     'MAPE:',MAPE,'MAPE_adjust:',MAPE_adjust,'sMAPE:',sMAPE,']')

predict=apt_rawdata.query("time> '2020-01-09 00:00' and time<= '2020-01-16 00:00'")[['calib_usage']]
predict['calib_usage_1dayago']=apt_rawdata[['calib_usage']].shift(24).query("time> '2020-01-09 00:00' and time<= '2020-01-16 00:00'")
predict['calib_usage_1weekago']=apt_rawdata[['calib_usage']].shift(168).query("time> '2020-01-09 00:00' and time<= '2020-01-16 00:00'")

predict[['calib_usage', 'calib_usage_1dayago']].plot(figsize=(20, 10))
predict[['calib_usage', 'calib_usage_1weekago']].plot(figsize=(20, 10))
precision(predict,'calib_usage_1dayago','calib_usage')
precision(predict,'calib_usage_1weekago','calib_usage')



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

data = apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'].values
n_test = 48
cfg_list = exp_smoothing_configs(seasonal=[24]) #[0,6,12,24]
scores = grid_search(data, cfg_list, n_test)
for cfg, error in scores[:3]:
	print('Best ExponentialSmoothing%s RMSE=%.3f' % (cfg, error))

# trend변수에 'add'를 설정할 경우 트랜드가 직선으로 증가 혹은 감소, 'mul'를 설정할 경우 트랜드가 지수적으로 증가 혹은 감소
ets_model = ExponentialSmoothing(apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'], trend=None, damped=False, seasonal='add', seasonal_periods=24)\
.fit(optimized=True, use_boxcox=False, remove_bias=True) #APT
ets_model = ExponentialSmoothing(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'], trend=None, damped=False, seasonal='add', seasonal_periods=24)\
.fit(optimized=True, use_boxcox=False, remove_bias=False) #APT
ets_model = ExponentialSmoothing(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'], trend='add', damped=False, seasonal='add', seasonal_periods=24)\
.fit(optimized=True, use_boxcox=False, remove_bias=False) #APT

ets_model = ExponentialSmoothing(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'], trend='add', damped=False, seasonal='add', seasonal_periods=24)\
.fit(optimized=True, use_boxcox=False, remove_bias=True) # Hotel

ets_model = ExponentialSmoothing(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'], trend=None, damped=False, seasonal='add', seasonal_periods=24)\
.fit(optimized=True, use_boxcox=False, remove_bias=True) # Hands6 

predict['forecast_ets'] =list( ets_model.predict(start = 0, end= 167)  )
predict['forecast_ets']=np.where(predict['forecast_ets']<20 ,0,predict['forecast_ets'])
predict[['calib_usage', 'forecast_ets']].plot(figsize=(20, 10))
precision(predict,'forecast_ets','calib_usage')

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
 
data = apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'].values
n_test = 48	# data split
cfg_list = sarima_configs(seasonal=[0, 12, 24]) # model configs
scores = grid_search(data, cfg_list, n_test)# grid search
	
# list top 3 configs
for cfg, error in scores[:5]:
	print('Best SARIMA%s MAE=%.3f' % (cfg, error))
    
sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(0,1,1,24),trend='n').fit()#apt
sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'],order=(1,0,2), seasonal_order=(1,1,0,24),trend='n').fit()# apt

sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(1,1,2,24),trend='ct').fit()

sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(1,0,1,24),trend='n').fit()
sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(0,1,1,24),trend='n').fit()
sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(1,1,0,24),trend='n').fit() # hotel
sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'],order=(0,0,0), seasonal_order=(1,1,0,168),trend='n').fit() # hotel

sarima_model = sm.tsa.SARIMAX(apt_rawdata.query("time<= '2018-12-15 00:00'")['calib_usage'],order=(0,1,0), seasonal_order=(1,1,0,24),trend='n').fit() # hands6

sarima_model.plot_diagnostics(figsize=(20, 10));

predict['forecast_sarima'] =list( sarima_model.predict(start =24, end= 191)  )
predict['forecast_sarima']=np.where(predict['forecast_sarima']<0 ,0,predict['forecast_sarima'])
predict[['calib_usage', 'forecast_sarima']].plot(figsize=(20, 10))
precision(predict,'forecast_sarima','calib_usage')


#building the model
from pmdarima.arima import auto_arima
arima_model = auto_arima(apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'], trace=True, start_p=0, d=None, strat_q=0, max_p=2, max_d=2, max_q=2,
                         start_P=0, D=None, strat_Q=0, max_P=2, max_D=2, max_Q=2, max_order=4,m=24, seasonal=True, infomation_criterion='aic', scoring='mse', suppress_warnings=True)\
                         .fit(apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage']) #apt

predict['forecast_arima'] = arima_model.predict(168)
predict['forecast_arima']=np.where(predict['forecast_arima']<20 ,0,predict['forecast_arima'])
predict[['calib_usage', 'forecast_arima']].plot(figsize=(20, 10))
precision(predict,'forecast_arima','calib_usage')


# day and week season
tbats_model = TBATS(seasonal_periods=(1, 24)).fit(apt_rawdata.query("time<= '2020-01-09 00:00'")['calib_usage'])

predict['forecast_tbats'] = tbats_model.forecast(steps=168)
predict['forecast_tbats']=np.where(predict['forecast_tbats']<20 ,0,predict['forecast_tbats'])
predict[['calib_usage', 'forecast_tbats']].plot(figsize=(20, 10))
precision(predict,'forecast_tbats','calib_usage')





#validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_squared_error(predict['calib_usage'], predict['calib_usage_1dayago'])
mean_squared_error(predict['calib_usage'], predict['calib_usage_1weekago'])
mean_squared_error(predict['calib_usage'], predict['forecast_ets'])
mean_squared_error(predict['calib_usage'], predict['forecast_arima'])
mean_squared_error(predict['calib_usage'], predict['forecast_sarima'])
mean_squared_error(predict['calib_usage'], predict['forecast_tbats'])

mean_absolute_error(predict['calib_usage'], predict['calib_usage_1dayago'])
mean_absolute_error(predict['calib_usage'], predict['calib_usage_1weekago'])
mean_absolute_error(predict['calib_usage'], predict['forecast_ets'])
mean_absolute_error(predict['calib_usage'], predict['forecast_arima'])
mean_absolute_error(predict['calib_usage'], predict['forecast_sarima'])
mean_absolute_error(predict['calib_usage'], predict['forecast_tbats'])

mean_absolute_percentage_error(predict['calib_usage'], predict['forecast_ets'])
round(mean_absolute_percentage_error(predict['calib_usage'], predict['calib_usage_1dayago']),2)
round(mean_absolute_percentage_error(predict['calib_usage'], predict['calib_usage_1weekago']),2)
                               



predict=apt_rawdata.query("time>'2019-02-01 00:00'")[['calib_usage']]
predict['calib_usage_1dayago']=apt_rawdata[['calib_usage']].shift(24)
predict['calib_usage_1weekago']=apt_rawdata[['calib_usage']].shift(168)




score_model1=['calib_usage_1dayago','calib_usage_1weekago','forecast_ets','forecast_arima','forecast_sarima','forecast_tbats']

for i in score_model1:
    precision(predict,i,'calib_usage')
    
    
    
predict= apt_rawdata.query("time> '2019-11-01 00:00'")
# validate
from tscv import GapWalkForward
cv = GapWalkForward(n_splits=10, gap_size=0, test_size=48)
for train_index, test_index in cv.split(predict):
#    print("train_index:", train_index, "test_index:", test_index)
    train, test =apt_rawdata.iloc[train_index,:], apt_rawdata.iloc[test_index,:]
    ets_model = ExponentialSmoothing(train, trend=None, damped=False, seasonal='add', seasonal_periods=24)\
    .fit(optimized=True, use_boxcox=False, remove_bias=True) #APT
    test['forecast_ets'] =list( ets_model.predict(start = 0, end= len(test)-1)  )
    test['forecast_ets']=np.where(test['forecast_ets']<0,0,test['forecast_ets'])
    print(precision(test,'forecast_ets','calib_usage'))