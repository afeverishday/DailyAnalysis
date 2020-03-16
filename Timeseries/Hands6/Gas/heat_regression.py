#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:45:48 2020

@author: keti
"""

# Regression=============================================================================================================
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
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage','보정계수':'calib_const', '순간유량': 'flux',
                  '압력':'pressure', '온도':'temp'})\
                      .assign(time=lambda x: pd.to_datetime(x['time']),
                              설비번호=lambda x: x['설비번호'].astype('str'),
                              설치번호=lambda x: x['설치번호'].astype('str'),
                              납부자번호=lambda x: x['납부자번호'].astype('str'),
                              ID=lambda x: x['ID'].astype('str'),                                        
                              calib_usage=lambda x: x['calib_usage'].str.replace(',','').astype('float'),
                              raw_usage=lambda x: x['raw_usage'].str.replace(',','').astype('float'))\
                              .query("time> '2017-01-01 00:00' and time<= '2020-01-29 00:00'")\
                              .drop(['ID', 'calib_const', 'flux', 'gas_type', '주소',  '고객명', '납부자번호', '설비번호', '설치번호',], axis=1)

weather = pd.DataFrame(pd.read_csv('hands6/수원기상_20170101.csv'))\
            .rename(columns={'일시' : 'time','지점':'station_num',  '지점명':'station_name', '기온(°C)':'temp', '풍속(m/s)':'windv', 
                             '습도(%)':'hum', '일사(MJ/m2)':'radiation','전운량(10분위)':'cloud', '지면온도(°C)':'land_temp'})\
            .assign(time=lambda x: pd.to_datetime(x['time']),
                    station_num=lambda x: x['station_num'].astype('str'),
                    station_name=lambda x: x['station_name'].astype('str'),
                    temp=lambda x: x['temp'].astype('float'),
                    hum=lambda x: x['hum'].astype('float'))[['time','station_name','temp','hum' ]]\
            .query("time> '2017-01-01 00:00' and time<= '2020-01-29 00:00'")


# replace
len(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(rawdata.time))
len(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(weather.time))

diff_rawdata = pd.DataFrame( columns=['time'] )
diff_rawdata['time'] = list(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(rawdata.time))
diff_rawdata['calib_usage']=None
rawdata=pd.concat([rawdata, diff_rawdata],axis=0)
rawdata=rawdata.sort_values(by='time').reset_index(drop=True)

diff_weather = pd.DataFrame( columns=['time'] )
diff_weather['time'] = list(set(pd.date_range(start ='2017-01-01 01:00:00', end = '2020-01-29 00:00:00', freq = 'H'))-set(weather.time))
diff_weather['temp']=None
diff_weather['hum']=None
diff_weather['station_name']='수원'
weather=pd.concat([weather, diff_weather],axis=0)
weather=weather.sort_values(by='time').reset_index(drop=True)

del(diff_rawdata, diff_weather)
gc.collect()

rawdata =rawdata.assign(year=lambda x: x['time'].dt.year.astype('str'),
                        month=lambda x: x['time'].dt.month.astype('str'),
                        hour=lambda x: x['time'].dt.hour.astype('str'),
                        week= lambda x: x['time'].dt.weekday_name)
                                                                     
rawdata.isnull().sum()
rawdata.info()
rawdata.nunique()


#결측된 시간 1주일 전 동 시간대 데이터로 채움
for i in list(set(rawdata.columns)-set(['flux','pressure','temp' ])):
    rawdata[i] = np.where(pd.notnull(rawdata[i]) == True, rawdata[i], rawdata[i].shift(168))

#이상치 데이터 1주일 전 동 시간대 데이터로 채움
for i in list(set(['calib_usage', 'raw_usage','pressure','temp' ])):
        rawdata[i] = np.where( (rawdata[i]< 500) & (rawdata[i] > 0) , rawdata[i], 
               np.where(pd.notnull(rawdata[i].shift(168))==True, rawdata[i].shift(168), rawdata[i]))

for i in list(set(['pressure','temp' ])):
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
del(holiday, vacation)

# holiday 데이터 1주일 전 동 시간대 데이터로 채움
for i in list(set(['calib_usage', 'raw_usage' ])):
    rawdata[i] = np.where( rawdata['holiday']==False , rawdata[i],
           np.where(rawdata['holiday'].shift(168)==False, rawdata[i].shift(168), rawdata[i].shift(336)))
#    rawdata[i] = np.where(rawdata['holiday_1dayago']==False & (rawdata['hour'].isin(range(18,24))),rawdata[i], 
#           np.where(pd.notnull(rawdata[i].shift(168))==True, rawdata[i].shift(168), rawdata[i]))
#    rawdata[i] = np.where(rawdata['holiday_1dayafter']==False & (rawdata['hour'].isin(range(0,6))),rawdata[i], 
#           np.where(pd.notnull(rawdata[i].shift(168))==True, rawdata[i].shift(168), rawdata[i]))

# columns name sort
numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ]))
categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"]))
time_feature = list(set(rawdata.columns) - set(categorical_feature)-set(numerical_feature))

# Categorical Feature
for col in categorical_feature:
    unique_list = rawdata[col].unique()
    
    print(unique_list)

#from natsort import natsorted
#for col in categorical_feature:
##    rawdata[col].value_counts().plot(kind='bar')
#    rawdata[col].value_counts().plot.pie(autopct='%.2f%%')
##		print(rawdata.groupby(col).mean())
#    print(rawdata.groupby(col).mean().reindex(index=natsorted(rawdata.groupby(col).mean().index)))
#    plt.title(col)
#    plt.show()


#rawdata['calib_usage_1dayago']= rawdata['calib_usage'].shift(24)
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

#check the replace
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

# describe
rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2017-01-01 00:00' and time<= '2017-05-18 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2017-05-18 00:00' and time<= '2017-08-10 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2017-08-10 00:00' and time<= '2018-09-03 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2018-09-03 00:00' and time<= '2020-01-05 00:00'")[numerical_feature].describe().round(2).transpose()
rawdata.query("time> '2020-01-05 00:00' and time<= '2020-01-29 00:00'")[numerical_feature].describe().round(2).transpose()


data = rawdata1.query("time> '2017-01-09 00:00' and time<= '2017-05-08 00:00'").assign(time= lambda x: pd.to_datetime(x.index))\
.drop(['year', 'month', 'holiday', 'raw_usage' ],axis=1)


import collections
collections.Counter(data['week'])
collections.Counter(data['hour'])


# find best parameter of model ============================================================================================================

data.isnull().sum()
data.info()
data.nunique()


data_mean=data[['week','hour','calib_usage']].groupby(['week','hour']).mean().reset_index()
data_mean.columns=['week','hour', 'forecast_mean']
data=pd.merge(data, data_mean, on=['week','hour']).sort_values(['time']).set_index(['time'])


data_dummy= data[['week','hour']]
data= data.drop(['week','hour'], axis=1)
data_dummy=pd.get_dummies(data_dummy)
data=pd.concat([data,data_dummy], axis=1)



#GridSearchCV===============================================
from sklearn.model_selection import GridSearchCV
from tscv import GapWalkForward
#  n_jobs= 4, 병렬 처리갯수? -1은 전부),   refit=True 좋은 estimator로 수정되어짐.
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
#    alpha : float, optional
#         디폴트 1.0,   클수록 가중치 규제 (특성 수 줄이기, 과대적합 방지),   작을수록 정확하게 (과대적합)
#    alpha 0이면 ordinary least square (일반적 리니어리그래션)
#           가중치가 커질수록 특성이 늘어나 훈련 데이터에 과적합됨   alpha 옵션을 크게 할수록 가중치 규제 (가중치의 크기가 못 커지게 하기, 과적합 방지)
#           크기 개념을 l1(맨하튼 거리) 으로   하냐 l2(유클리디안 거리의 제곱)로 생각하냐에 따라  라쏘(l1) 혹은 릿지(l2) 를 사용
param_grid = { 'alpha':np.arange(0,0.1,0.01) }
lasso_model = GridSearchCV( Lasso(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168), scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(lasso_model.best_params_)
print(lasso_model.best_score_)

param_grid = { 'alpha':np.arange(0,1,0.1) }
ridge_model = GridSearchCV( Ridge(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(ridge_model.best_params_)
print(ridge_model.best_score_)

param_grid = { 'alpha':np.arange(0,1,0.1), 'l1_ratio':np.arange(0,1,0.1) }
elasticnet_model = GridSearchCV( ElasticNet(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168), scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(elasticnet_model.best_params_)
print(elasticnet_model.best_score_)

#from sklearn.neural_network import MLPRegressor
#param_grid = {'max_iter':np.arange(5,100,5), 'alpha':np.arange(0.1,1,0.1),'hidden_layer_sizes':np.arange(10,200,10)}
#mlp_model = GridSearchCV(MLPRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True).fit(train_x, train_y)
#print(mlp_model.best_params_)
#print(mlp_model.best_score_)

from sklearn.tree import DecisionTreeRegressor
#    criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'                     
param_grid = {'max_features':np.arange(1,5,1), 'max_depth':np.arange(1,15,1)}
tree_model = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop(['calib_usage'],axis=1), data[['calib_usage']])
print(tree_model.best_params_)
print(tree_model.best_score_)

from sklearn.svm import LinearSVR, SVR
#    kernel : linear, poly,rbf
#    C : float, optional (default=1.0) 클수록 정확하게 (마진이 작아짐, 과대적합),  alpha (가중치 규제) 의 역수
#    gamma : float, optional (default=’auto’) 클수록 정확하게 (경사가 급해짐, 과대적합)   비선형 (kernel=‘rbf’, kernel=‘poly’)에서만 이 옵션 사용
param_grid = {'kernel':['linear','rbf','poly'], 'C':np.arange(1,4,1)}
svr_model = GridSearchCV(SVR(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(svr_model.best_params_)
print(svr_model.best_score_)

from sklearn.ensemble import BaggingRegressor
#    base_estimator: base estimator to fit
#    n_estimators: the number of base estimators
#    bootstrap : (default=True) 중복 할당 여부  True 베깅, S  False 페이스팅
param_grid = {'base_estimator':[Ridge(alpha=0),LinearRegression(fit_intercept=True),Lasso(alpha=0)], 'n_estimators':np.arange(2,20,2),'bootstrap':[True, 'S',False]}
bagging_model = GridSearchCV(BaggingRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop(['calib_usage'],axis=1), data[['calib_usage']])
print(bagging_model.best_params_)
print(bagging_model.best_score_)

from sklearn.ensemble import RandomForestRegressor
#    n_estimators : integer, optional (default=10) The number of trees in the forest.
#    bootstrap : boolean, optional (default=True) True 베깅,   False 페이스팅
#    criterion : string, optional (default=”mse”) 'mse' (평균제곱오차),   'friedman_mse', 'mae'
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'
param_grid = {'n_estimators':np.arange(2,20,2),'bootstrap':[True,False],'max_features':np.arange(1,5,1), 'max_depth':np.arange(1,15,1)}
rf_model = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(rf_model.best_params_)
print(rf_model.best_score_)

from sklearn.ensemble import AdaBoostRegressor
#    base_estimator : object, optional (default=None)
#    n_estimators : integer, optional (default=50) The maximum number of estimators at which boosting is terminated.
param_grid = {'n_estimators':np.arange(10,100,10),'base_estimator':[Ridge(alpha=0),LinearRegression(fit_intercept=True),Lasso(alpha=0)]}
adaboost_model = GridSearchCV(AdaBoostRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(adaboost_model.best_params_)
print(adaboost_model.best_score_)

from sklearn.ensemble import GradientBoostingRegressor
#    n_estimators : int (default=100) The number of boosting stages to perform.
#    criterion : string, optional (default=”friedman_mse”) friedman_mse, mse, mae
param_grid = {'n_estimators':np.arange(50,150,10)}
gradient_model = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(gradient_model.best_params_)
print(gradient_model.best_score_)

from xgboost import XGBRegressor
# booster: 의사결정 기반 모형(gbtree), 선형 모형(linear)
# mthread: 병렬처리에 사용되는 코어수, 특정값을 지정하지 않는 경우 자동으로 시스템 코어수를 탐지하여 병렬처리에 동원함.
# eta [기본설정값: 0.3]: GBM에 학습율과 유사하고 일반적으로 0.01 ~ 0.2 값이 사용됨
# min_child_weight [기본설정값: 1]: 과적합(overfitting)을 방지할 목적으로 사용되는데, 너무 높은 값은 과소적합(underfitting)을 야기하기 때문에 CV를 사용해서 적절한 값이 제시되어야 한다.
# max_depth [기본설정값: 6]: 과적합 방지를 위해서 사용되는데 역시 CV를 사용해서 적절한 값이 제시되어야 하고 보통 3-10 사이 값이 적용된다.
# max_leaf_nodes: max_leaf_nodes 값이 설정되면 max_depth는 무시된다. 따라서 두값 중 하나를 사용한다.
# max_delta_step [기본설정값: 0]: 일반적으로 잘 사용되지 않음.
# subsample [기본설정값: 1]: 개별 의사결정나무 모형에 사용되는 임의 표본수를 지정. 보통 0.5 ~ 1 사용됨.
# colsample_bytree [기본설정값: 1]: 개별 의사결정나무 모형에 사용될 변수갯수를 지정. 보통 0.5 ~ 1 사용됨.
# colsample_bylevel [기본설정값: 1]: subsample, colsample_bytree 두 초모수 설정을 통해서 이미 의사결정나무 모형 개발에 사용될 변수갯수와 관측점 갯수를 사용했는데 추가로 colsample_bylevel을 지정하는 것이 특별한 의미를 갖는지 의문이 듦.
# lambda [기본설정값: 1]: 능선 회쉬(Ridge Regression)의 L2 정규화(regularization) 초모수. 그다지 많이 사용되고 있지는 않음.
# alpha [기본설정값: 0]: 라쏘 회귀(Lasso Regression)의 L1 정규화(regularization) 초모수로 차원이 높은 경우 알고리즘 속도를 높일 수 있음.
# scale_pos_weight [기본설정값: 1]: 클래스 불균형이 심한 경우 0보다 큰 값을 지정하여 효과를 볼 수 있음.
param_grid = {'eta':np.arange(0.001,0.01,0.001),'max_depth':np.arange(1,10,1)}
xgb_model = GridSearchCV(XGBRegressor(objective ='reg:squarederror'), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(xgb_model.best_params_)
print(xgb_model.best_score_)

from lightgbm import LGBMRegressor
param_grid = {'n_estimators':np.arange(10,50,10),'learning_rate':np.arange(0.001,0.01,0.001)}
lgb_model = GridSearchCV(LGBMRegressor(), param_grid=param_grid, cv=GapWalkForward(n_splits=5, gap_size=0, test_size=168),scoring='r2',refit=True, n_jobs=2)\
.fit(data.drop([ 'calib_usage'],axis=1), data[['calib_usage']])
print(lgb_model.best_params_)
print(lgb_model.best_score_)













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






Total= pd.DataFrame(columns=['time','calib_usage', 'calib_usage_1weekago', 'forecast_mean','model_average', 'forecast_ridge',
                             'forecast_rf','forecast_xgb'])
result=pd.DataFrame(columns=['Rsquared', 'MAE', 'MSE', 'RMSE', 'MSLE', 'RMLSE','MAPE', 'MAPE_adjust','sMAPE','Model'])
# model evaluation============================================================================================================
from tscv import GapWalkForward
from sklearn.model_selection import cross_val_score
cv = GapWalkForward(n_splits=5, gap_size=0, test_size=168)

for train, test in cv.split(data):
    train_x, train_y =data.iloc[train].drop(['calib_usage'], axis=1) , data.iloc[train].calib_usage
    test_x, test_y =data.iloc[test].drop(['calib_usage'], axis=1) , data.iloc[test].calib_usage
    
    train_y=pd.DataFrame(train_y)
    train=pd.concat([train_x, train_y], axis=1) # columns bind
    train_mean=train[['week','hour','calib_usage']].groupby(['week','hour']).mean().reset_index()
    train_mean.columns=['week','hour', 'forecast_mean']
    train=pd.merge(train, train_mean, on=['week','hour']).sort_values(['time']).set_index(['time'])
    train_x=train.drop(['calib_usage'], axis=1)

    test=pd.concat([test_x, test_y], axis=1) # columns bind
    test=pd.merge(test, train_mean, on=['week','hour']).sort_values(['time']).set_index(['time'])
    test_x=test.drop(['calib_usage'], axis=1)
    
    # columns name sort
    categorical_feature = list( set([ col for col in train.columns if train[col].dtypes == "object"]))
    numerical_feature = list( set([ col for col in train.columns if train[col].dtypes in(['float64', 'int64']) ]))
    
    minmax_feature = list( set(numerical_feature)-set(['calib_usage', 'temp']))
    standard_feature = list(set(['temp']))
    target_feature = list(['calib_usage'])
    dummy_feature=list(set(categorical_feature))

    # scaler
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    minmaxscaler = MinMaxScaler().fit(train_x[minmax_feature])
    minmaxscaler_target = MinMaxScaler().fit(train_y)
    standardscaler = StandardScaler().fit(train_x[standard_feature])
    train_minmax = pd.DataFrame(minmaxscaler.transform(train_x[minmax_feature]),
                                columns= minmax_feature, index=list(train_x.index.values))
    train_target = pd.DataFrame(minmaxscaler_target.transform(train_y),
                                columns= target_feature, index=list(train_y.index.values))
    train_standard = pd.DataFrame(standardscaler.transform(train_x[standard_feature]),
                                columns= standard_feature, index=list(train_x.index.values))
    train_dummy=pd.get_dummies(train_x[categorical_feature], prefix=categorical_feature) #, drop_first=True
    
    train_preprocess=pd.concat([train_minmax, train_standard,train_dummy], axis=1) # columns bind
    train_all=pd.concat([train_preprocess, train_target], axis=1)

    train_x= train_preprocess
    train_y= train_target
    ridge_model=Ridge(alpha= 0.9).fit(train_x, train_y)
#    tree_model=DecisionTreeRegressor(max_depth= 6, max_features= 4).fit(train_x, train_y)
    rf_model=RandomForestRegressor(bootstrap= True, max_depth=13, max_features= 3, n_estimators= 18).fit(train_x, train_y)
    xgb_model=XGBRegressor(objective ='reg:squarederror',eta= 0.001, max_depth= 1).fit(train_x, train_y)
#    lgb_model=LGBMRegressor(learning_rate= 0.009, n_estimators= 40).fit(train_x, train_y)

    test_minmax = pd.DataFrame(minmaxscaler.transform(test_x[minmax_feature]),
                                columns= minmax_feature, index=list(test_x.index.values))
    test_standard = pd.DataFrame(standardscaler.transform(test_x[standard_feature]),
                                columns= standard_feature, index=list(test_x.index.values))
    test_target = pd.DataFrame(minmaxscaler_target.transform(pd.DataFrame(test_y)),
                                columns= target_feature, index=list(test_y.index.values))
    test_dummy=pd.get_dummies(test_x[categorical_feature], prefix=categorical_feature) #, drop_first=True
       
    test_x=pd.concat([test_minmax, test_standard, test_dummy], axis=1) # columns bind
    test_y=test_target
    test_all=pd.concat([test_x, test_y], axis=1)
    
    predict= test_all[['calib_usage', 'calib_usage_1weekago', 'forecast_mean']]\
    .assign(time= lambda x: pd.to_datetime(x.index),
            calib_usage=lambda x: minmaxscaler_target.data_max_*(x['calib_usage'])+minmaxscaler_target.data_min_,
            calib_usage_1weekago=lambda x: minmaxscaler_target.data_max_*(x['calib_usage_1weekago'])+minmaxscaler_target.data_min_,
            forecast_mean=lambda x: minmaxscaler_target.data_max_*(x['forecast_mean'])+minmaxscaler_target.data_min_,
            forecast_ridge= minmaxscaler_target.data_max_*np.where(ridge_model.predict(test_x).astype('float')<0,0,ridge_model.predict(test_x).astype('float'))+minmaxscaler_target.data_min_,
            forecast_rf= minmaxscaler_target.data_max_*np.where(rf_model.predict(test_x).astype('float')<0,0,rf_model.predict(test_x).astype('float'))+minmaxscaler_target.data_min_,
            forecast_xgb= minmaxscaler_target.data_max_*np.where(xgb_model.predict(test_x).astype('float')<0,0,xgb_model.predict(test_x).astype('float'))+minmaxscaler_target.data_min_,
            model_average= lambda x: x[['forecast_ridge', 'forecast_rf', 'forecast_xgb']].mean(axis=1))


    Total=pd.concat([Total, predict], axis=0)
    
    result=pd.concat([result,precision(predict,'forecast_mean', 'calib_usage').assign(Model='Mean')], axis=0 )\
             .append(precision(predict,'calib_usage_1weekago', 'calib_usage').assign(Model='Naive_model'))\
             .append(precision(predict,'forecast_ridge', 'calib_usage').assign(Model='Ridge_model'))\
             .append(precision(predict,'forecast_rf', 'calib_usage').assign(Model='RF_model'))\
             .append(precision(predict,'forecast_xgb', 'calib_usage').assign(Model='XGB_model'))\
             .append(precision(predict,'model_average', 'calib_usage').assign(Model='Model_Average'))
    


# show the result
result=result.set_index('Model')
print(result.apply(lambda x: x.astype('float')).reset_index().groupby(['Model']).mean())




