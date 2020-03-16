#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:45:32 2020

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


rawdata = pd.DataFrame(pd.read_csv('hands6/heat/핸즈식스2공장_3번.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage','보정계수':'calib_const', '순간유량': 'flux'})\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                                                                      설비번호=lambda x: x['설비번호'].astype('str'),
                                                                      설치번호=lambda x: x['설치번호'].astype('str'),
                                                                      납부자번호=lambda x: x['납부자번호'].astype('str'),
                              ID=lambda x: x['ID'].astype('str'),                                        
                              calib_usage=lambda x: x['calib_usage'].str.replace(',','').astype('float'),
                              raw_usage=lambda x: x['raw_usage'].str.replace(',','').astype('float'))\
                              .query("time> '2017-01-01 00:00' and time<= '2020-01-29 00:00'")
   # replace
len(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(rawdata.index))
diff = pd.DataFrame( columns=['time'] )
diff['time'] = list(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(rawdata.time))
diff['calib_usage']=None
rawdata=pd.concat([rawdata, diff],axis=0)
rawdata=rawdata.sort_values(by='time')

rawdata =rawdata.assign(year=lambda x: x['time'].dt.year.astype('str'),
                        month=lambda x: x['time'].dt.month.astype('str'),
                        hour=lambda x: x['time'].dt.hour.astype('str'),
                        week= lambda x: x['time'].dt.weekday_name)
                                                                     
rawdata.isnull().sum()
rawdata.info()
rawdata.nunique()

# Categorical Feature
for col in categorical_feature:
    unique_list = rawdata[col].unique()
    
    print(unique_list)

from natsort import natsorted
for col in categorical_feature:
#    rawdata[col].value_counts().plot(kind='bar')
    rawdata[col].value_counts().plot.pie(autopct='%.2f%%')
#		print(rawdata.groupby(col).mean())
    print(rawdata.groupby(col).mean().reindex(index=natsorted(rawdata.groupby(col).mean().index)))
    plt.title(col)
    plt.show()




#결측된 시간 1주일 전 동 시간대 데이터로 채움
for i in list(set(rawdata.columns)-set(['flux','압력','온도' ])):
    rawdata[i] = np.where(pd.notnull(rawdata[i]) == True, rawdata[i], rawdata[i].shift(168))

#이상치 데이터 1주일 전 동 시간대 데이터로 채움
for i in list(set(['calib_usage', 'raw_usage','압력','온도' ])):
    rawdata[i] = np.where( (rawdata[i]< 500) & (rawdata[i] > 0) , rawdata[i], 
           np.where(pd.notnull(rawdata[i].shift(168))==True, rawdata[i].shift(168), rawdata[i]))

for i in list(set(['압력','온도' ])):
    rawdata[i] = np.where( rawdata[i] > 0 , rawdata[i], 
           np.where(pd.notnull(rawdata[i].shift(1))==True, rawdata[i].shift(1), rawdata[i].shift(24)))

# holiday
import requests
import pandas as pd

def get_request_query(url, operation, params, serviceKey):
    import urllib.parse as urlparse
    params = urlparse.urlencode(params)
    request_query = url + '/' + operation + '?' + params + '&' + 'serviceKey' + '=' + serviceKey+'&_type=json'
    return request_query

# 요청 URL과 오퍼레이션
URL = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService'
OPERATION = 'getRestDeInfo' # 국경일 + 공휴일 정보 조회 오퍼레이션
SERVICEKEY = 'f0zYRYA98oJ0kjZpHkrHyzOMbBXmY7Iwev8c8n35kw%2FFlpgBHtsVTb6aD%2BKIPUgo3g2BAUisHDuSDNF7wLaZ%2Bg%3D%3D'# 파라미터
PARAMS = {'solYear':'2017', 'solMonth':'01'}

holiday=pd.DataFrame(columns=['dateKind', 'dateName', 'isHoliday', 'locdate', 'seq'])

for year in range(2017,2021):
    print(year)
    for month in range(1,13):
        if month<10:
            PARAMS = {'solYear':str(year), 'solMonth': '0'+str(month)}
            print(PARAMS)
        else:
            PARAMS = {'solYear':str(year), 'solMonth': str(month)}
            print(PARAMS)
        request_query = get_request_query(URL, OPERATION, PARAMS, SERVICEKEY)
        html= requests.get(request_query)
        dictr=html.json().get('response').get('body').get('items')

        if dictr !=  '':
            recs = dictr['item']
            from pandas.io.json import json_normalize
            df = json_normalize(recs)
            holiday=pd.concat([holiday, df], axis=0)

del(year, month, dictr, recs, df, request_query)

holiday=holiday.assign(date= pd.to_datetime(holiday['locdate'].astype(str)).dt.date).drop(['dateKind', 'isHoliday','locdate','seq' ], axis=1)

vacation= pd.DataFrame(data=pd.to_datetime(pd.date_range(start='2017-07-31',end='2017-08-06', freq ='1D')
                       .append(pd.date_range(start='2018-08-02',end='2018-08-10', freq ='1D'))
                       .append(pd.date_range(start='2019-08-05',end='2019-08-10', freq ='1D'))) , columns=['date']).assign(dateName='휴가')

holiday= pd.concat([holiday, vacation], axis=0).sort_values(by=['date'], axis=0).reset_index(drop=True)
holiday=holiday.assign(date= pd.to_datetime(holiday['date'].astype(str)).dt.date)

from datetime import datetime, timedelta 
rawdata['holiday'] = np.where( rawdata['time'].dt.date.isin(holiday['date'])==False , False,True) 
#rawdata['holiday_1dayago'] = np.where( rawdata['time'].dt.date.isin(holiday['date']-timedelta(days=1))==False , 0,1) 
#rawdata['holiday_1dayafter'] = np.where( rawdata['time'].dt.date.isin(holiday['date']+timedelta(days=1))==False , 0,1) 


# holiday 데이터 1주일 전 동 시간대 데이터로 채움
for i in list(set(['calib_usage', 'raw_usage' ])):
    rawdata[i] = np.where( rawdata['holiday']==False , rawdata[i],
           np.where(rawdata['holiday'].shift(168)==False, rawdata[i].shift(168), rawdata[i].shift(336)))
#    rawdata[i] = np.where(rawdata['holiday_1dayago']==False & (rawdata['hour'].isin(range(18,24))),rawdata[i], 
#           np.where(pd.notnull(rawdata[i].shift(168))==True, rawdata[i].shift(168), rawdata[i]))
#    rawdata[i] = np.where(rawdata['holiday_1dayafter']==False & (rawdata['hour'].isin(range(0,6))),rawdata[i], 
#           np.where(pd.notnull(rawdata[i].shift(168))==True, rawdata[i].shift(168), rawdata[i]))


rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
rawdata['calib_usage_1weekago']= rawdata['calib_usage'].shift(168)


rawdata =rawdata.set_index('time')
rawdata1=rawdata.query("(time> '2017-01-01 00:00' and time<= '2017-05-18 00:00')" )
rawdata2=rawdata.query("(time> '2017-05-18 00:00' and time<= '2017-08-10 00:00')" )
rawdata3=rawdata.query("(time> '2017-08-10 00:00' and time<= '2018-05-21 00:00')" )
rawdata4=rawdata.query("(time> '2018-05-21 00:00' and time<= '2018-09-03 00:00')" )
rawdata4=rawdata.query("(time> '2018-06-30 00:00' and time<= '2018-08-15 00:00')" )
rawdata5=rawdata.query("(time> '2018-09-03 00:00' and time<= '2020-01-05 00:00')" )
rawdata5=rawdata.query("(time> '2019-07-03 00:00' and time<= '2019-08-15 00:00')" )
rawdata6=rawdata.query("(time> '2020-01-05 00:00' and time<= '2020-01-29 00:00')" )

rawdata1[['calib_usage']].plot()
rawdata2[['calib_usage']].plot()
rawdata3[['calib_usage']].plot()
rawdata4[['calib_usage']].plot()
rawdata5[['calib_usage']].plot()
rawdata6[['calib_usage']].plot()

def replace(data):
    for i in range(24):
        for j in data['week'].unique():
            #print(j)
            data.loc[(data.hour==str(i)) & (data.week==j),'calib_usage'] = data[(data.hour==str(i)) & (data.week==j)]['calib_usage']\
            .clip(min(0,data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].median() -3*data[(data.hour==str(i))&(data.week==j)]['calib_usage'].std() ),
                  data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].median() +3*data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].std() )
            print(data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].median() +3*data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].std())

replace(rawdata1)
replace(rawdata2)
replace(rawdata3)
replace(rawdata4)

def replace(data):
    for i in range(24):
        for j in data['week'].unique():
            #print(j)
            data.loc[(data.hour==str(i)) & (data.week==j),'calib_usage'] = data[(data.hour==str(i)) & (data.week==j)]['calib_usage']\
            .clip(data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].median() -3*data[(data.hour==str(i))&(data.week==j)]['calib_usage'].std() ,
                  data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].median() +3*data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].std() )
            print(data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].median() +3*data[(data.hour==str(i)) & (data.week==j)]['calib_usage'].std())

replace(rawdata5)
replace(rawdata6)

rawdata1[['calib_usage']].plot()
rawdata2[['calib_usage']].plot()
rawdata3[['calib_usage']].plot()
rawdata4[['calib_usage']].plot()
rawdata5[['calib_usage']].plot()
rawdata6[['calib_usage']].plot()
#for i in list(set(['calib_usage', 'raw_usage' ])):
#    rawdata[i] = np.where( (pd.to_datetime('2017-05-18 00:00')<rawdata['time'])& (rawdata['time']<= pd.to_datetime('2017-08-10 00:00')), 
#    rawdata[i]*1.55, rawdata[i])


#plt.subplot(411)
#plt.plot(rawdata.index, 
#         rawdata.calib_usage, linestyle='-', linewidth=2, color='g') # 'dashed'
#
#rawdata =rawdata.set_index('time')
#plt.figure() # 전체 그림을 리셋한다
#rawdata['calib_usage'].plot()

# usage for week& time
rawdata1.loc[rawdata1.hour==str(1)].query("(week in ('Sunday')) and (time> '2017-01-01 00:00' and time<= '2017-05-18 00:00')" )['calib_usage'].describe()
rawdata2.loc[rawdata2.hour==str(1)].query("(week in ('Sunday')) and (time> '2017-05-18 00:00' and time<= '2017-08-10 00:00')" )['calib_usage'].describe()
rawdata3.loc[rawdata3.hour==str(1)].query("(week in ('Sunday')) and (time> '2017-08-10 00:00' and time<= '2018-05-21 00:00')" )['calib_usage'].describe()
rawdata4.loc[rawdata4.hour==str(1)].query("(week in ('Sunday')) and (time> '2018-05-21 00:00' and time<= '2018-09-03 00:00')" )['calib_usage'].describe()
rawdata5.loc[rawdata5.hour==str(1)].query("(week in ('Sunday')) and (time> '2018-09-03 00:00' and time<= '2020-01-05 00:00')" )['calib_usage'].describe()
rawdata6.loc[rawdata6.hour==str(1)].query("(week in ('Sunday')) and ( time> '2020-01-05 00:00' and time<= '2020-01-29 00:00')" )['calib_usage'].describe()

# columns name sort
numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ]))
categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"]))
time_feature = list(set(rawdata.columns) - set(categorical_feature)-set(numerical_feature))

rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2017-05-18 00:00' and time<= '2017-08-10 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2017-08-10 00:00' and time<= '2018-09-03 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2018-09-03 00:00' and time<= '2020-01-05 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2020-01-05 00:00' and time<= '2020-01-29 00:00'")[numerical_feature].describe().round(2).transpose()

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


#gridsearch(ETS) start===============================================
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

data = rawdata1.query("time> '2017-01-02 00:00' and time<= '2017-05-15 00:00'")['calib_usage'].values
data = rawdata2.query("time> '2017-05-22 00:00' and time<= '2018-08-07 00:00'")['calib_usage'].values
data = rawdata3.query("time> '2017-08-14 00:00' and time<= '2018-05-21 00:00'")['calib_usage'].values
data = rawdata4.query("time> '2018-05-21 00:00' and time<= '2018-09-03 00:00'")['calib_usage'].values
data = rawdata5.query("time> '2018-09-03 00:00' and time<= '2019-12-30 00:00'")['calib_usage'].values
data = rawdata6.query("time> '2020-01-06 00:00' and time<= '2020-01-27 00:00'")['calib_usage'].values

# for rawdata1,2,3,4
n_test = 168
cfg_list = exp_smoothing_configs(seasonal=[ 168]) #[0,6,12,24,168]
scores = grid_search(data, cfg_list, n_test)

# for rawdata5,6
n_test = 168
cfg_list = exp_smoothing_configs(seasonal=[0,24, 168]) #[0,6,12,24,168]
scores = grid_search(data, cfg_list, n_test)

# list top 3 configs
for cfg, error in scores[:3]:
	print('Best ExponentialSmoothing%s RMSE=%.3f' % (cfg, error))

#gridsearch(ETS) end===============================================

# grid search(SARIMA) start===================================================================================================

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

data = rawdata1.query("time> '2017-01-02 00:00' and time<= '2017-05-15 00:00'")['calib_usage'].values
data = rawdata2.query("time> '2017-05-22 00:00' and time<= '2018-08-07 00:00'")['calib_usage'].values
data = rawdata3.query("time> '2017-08-14 00:00' and time<= '2018-05-21 00:00'")['calib_usage'].values
data = rawdata4.query("time> '2018-05-21 00:00' and time<= '2018-09-03 00:00'")['calib_usage'].values
data = rawdata5.query("time> '2018-09-03 00:00' and time<= '2019-12-30 00:00'")['calib_usage'].values
data = rawdata6.query("time> '2020-01-06 00:00' and time<= '2020-01-27 00:00'")['calib_usage'].values

# for rawdata1,2,3,4
n_test = 168# data split
cfg_list = sarima_configs(seasonal=[168]) # model configs
scores = grid_search(data, cfg_list, n_test)# grid search

# for rawdata5,6
n_test = 168# data split
cfg_list = sarima_configs(seasonal=[0, 24, 168]) # model configs
scores = grid_search(data, cfg_list, n_test)# grid search
	
# list top 3 configs
for cfg, error in scores[:5]:
	print('Best SARIMA%s MAE=%.3f' % (cfg, error))
    
    
# grid search(SARIMA) end===================================================================================================

resdiff=sm.tsa.arma_order_select_ic(data.iloc[train]['calib_usage'], max_ar=7, max_ma=7, ic='aic', trend='c')
print('ARMA(p,q) =',resDiff['aic_min_order'],'is the best.')


# Auto SARIMA start===================================================================================================
#building the model
from pmdarima.arima import auto_arima
arima_model1 = auto_arima(data1['calib_usage'], trace=True, start_p=0, d=None, strat_q=0, max_p=2, max_d=2, max_q=2,
                         start_P=0, D=None, strat_Q=0, max_P=2, max_D=2, max_Q=2, max_order=4,m=168, seasonal=True, infomation_criterion='aic', scoring='mse', suppress_warnings=True)\
                         .fit(data1['calib_usage']) 
predict3['forecast_arima'] = arima_model3.predict(168)
predict3['forecast_arima']=np.where(predict3['forecast_arima']<20 ,0,predict3['forecast_arima'])

# Auto SARIMA end===================================================================================================

# TBATS start===================================================================================================
#building the model
from pmdarima.arima import auto_arima
tbats_model2 = TBATS(seasonal_periods=(1, 24)).fit(data1['calib_usage'])
predict1['forecast_tbats'] = tbats_model1.forecast(steps=168)
predict1['forecast_tbats']=np.where(predict1['forecast_tbats']<20 ,0,predict1['forecast_tbats'])

# TBATS end===================================================================================================


#ETS model
data = rawdata1.query("time> '2017-01-09 00:00' and time<= '2017-05-15 00:00'")
data = rawdata2.query("time> '2017-05-22 00:00' and time<= '2017-08-07 00:00'")
data = rawdata3.query("time> '2017-08-14 00:00' and time<= '2018-05-21 00:00'")
data = rawdata4.query("time> '2018-05-21 00:00' and time<= '2018-09-03 00:00'")
data = rawdata5.query("time> '2018-09-03 00:00' and time<= '2019-12-30 00:00'")
data = rawdata6.query("time> '2020-01-06 00:00' and time<= '2020-01-27 00:00'")
# trend변수에 'add'를 설정할 경우 트랜드가 직선으로 증가 혹은 감소, 'mul'를 설정할 경우 트랜드가 지수적으로 증가 혹은 감소
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
     dict=pd.DataFrame({'Rsquared': Rsquared, 'MAE': MAE, 'MSE':MSE, 'RMSE':RMSE, 'MSLE': MSLE, 'RMLSE':RMLSE,\
           'MAPE': MAPE, 'MAPE_adjust': MAPE_adjust, 'sMAPE': sMAPE}.items()).transpose().drop(0)
     dict.columns=['Rsquared', 'MAE', 'MSE', 'RMSE', 'MSLE', 'RMLSE','MAPE', 'MAPE_adjust','sMAPE'] 
     return(dict)

from tscv import GapWalkForward
from sklearn.model_selection import cross_val_score
cv = GapWalkForward(n_splits=5, gap_size=0, test_size=168)

Total= pd.DataFrame(columns=['time','calib_usage', 'raw_usage', 'calib_usage_1dayago', 'calib_usage_1weekago','week','hour',
                             'forecast_ets','forecast_sarima','forecast_sarimax', 'forecast_mean'])
result=pd.DataFrame(columns=['Rsquared', 'MAE', 'MSE', 'RMSE', 'MSLE', 'RMLSE','MAPE', 'MAPE_adjust','sMAPE','Model'])
i=0
for train, test in cv.split(data):
    i=i+1
    #print("train:", data.iloc[train].calib_usage, "test:",data.iloc[test].calib_usage)
    ets_model = ExponentialSmoothing(data.iloc[train].calib_usage, 
                                 trend=None, damped=False, seasonal='add', seasonal_periods=168)\
                                 .fit(optimized=True, use_boxcox=False, remove_bias=False) # Hands6 
    sarima_model = sm.tsa.SARIMAX(data.iloc[train].calib_usage,
                                   order=(1,0,2), seasonal_order=(1,1,0,168),trend='n').fit()# hands6
    sarimax_model = SARIMAX(data.iloc[train].calib_usage, exog=data.iloc[train].calib_usage_1weekago, 
                        order=(1,0,2), seasonal_order=(1,1,0,168),trend='n').fit(disp=False)
    predict1= data.iloc[test][['calib_usage', 'raw_usage', 'calib_usage_1dayago', 'calib_usage_1weekago','week','hour']]\
    .assign(time= lambda x: x.index)
    predict1['forecast_ets'] =list(ets_model.predict(start = 0, end= len(predict1)-1)  ) 
    predict1['forecast_ets'] = np.where(predict1['forecast_ets']<0 ,0,predict1['forecast_ets'])
    predict1['forecast_sarima'] =list(sarima_model.predict(start = 168, end= 168+len(predict1)-1)  ) 
    predict1['forecast_sarima'] = np.where(predict1['forecast_sarima']<0 ,0,predict1['forecast_sarima'])
    predict1['forecast_sarimax'] =list(sarimax_model.predict(start = 168, end= 168+len(predict1)-1, exog=predict1['calib_usage_1weekago']) )
    predict1['forecast_sarimax'] = np.where(predict1['forecast_sarimax']<0 ,0,predict1['forecast_sarimax'])
    predict_mean=data.iloc[train][['week','hour','calib_usage']].groupby(['week','hour']).mean().reset_index()
    predict_mean.columns=['week','hour', 'forecast_mean']
    predict=pd.merge(predict1, predict_mean, on=['week','hour'])
    
    Total=pd.concat([Total, predict], axis=0)
#    predict1['forecast_simplemean'] = np.mean(data.iloc[test]['calib_usage'])
#    result=pd.concat([result,precision(predict1,'forecast_simplemean', 'calib_usage')], axis=0 )
    result=pd.concat([result,precision(predict,'forecast_mean', 'calib_usage').assign(Model='mean')], axis=0 )\
             .append(precision(predict,'calib_usage_1weekago', 'calib_usage').assign(Model='naive_model'))\
             .append(precision(predict,'forecast_ets', 'calib_usage').assign(Model='ETS_model'))\
             .append(precision(predict,'forecast_sarima', 'calib_usage').assign(Model='SRAIMA_model'))\
             .append(precision(predict,'forecast_sarimax', 'calib_usage').assign(Model='SRAIMAX_model'))
    if i==5:
        print(result.reset_index(drop=True))






















plt.figure() # 전체 그림을 리셋한다
rawdata =rawdata.set_index('time')
plt.subplot(411)
plt.plot(rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").index, 
         rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").calib_usage, linestyle='-', linewidth=2, color='r') # 'dashed'
plt.subplot(412)
plt.plot(rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").index, 
         rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").압력, linestyle='-', linewidth=2, color='g') # 'dashed'
plt.subplot(412)
plt.plot(rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").index, 
         rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").온도,  linestyle='-', linewidth=2, color='b') # 'dashed'

plt.plot(rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").index, 
         rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'").calib_usage, linestyle='-', linewidth=2, color='r') # 'dashed'


rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'")[['압력','온도', 'calib_usage']].plot()
























