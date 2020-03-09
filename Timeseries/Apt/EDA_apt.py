#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:49:38 2020

@author: keti
"""

import pandas as pd
import numpy as np

import gc
gc.collect()
import sys, os
os.getcwd()
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')

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
                              flux=lambda x: x['flux'].astype('float')).drop(['납부자번호', '고객명', 'ID', '주소', 'gas_type', '설치번호', '설비번호'],axis=1)\
                              .query("time>= '2018-01-01 00:00' and time< '2020-01-01 00:00'")


apt_rawdata = pd.DataFrame(pd.read_csv('hands6/heat/핸즈식스2공장_3번.csv'))\
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
                              flux=lambda x: x['flux'].astype('float')).drop(['납부자번호', '고객명', 'ID', '주소', 'gas_type', '설치번호', '설비번호'],axis=1)\
                              .query("time>= '2017-01-01 01:00' and time< '2020-01-29 00:00'")


apt_rawdata = pd.DataFrame( pd.read_csv('Apt/춘의주공APT.csv'))\
.rename(columns={'검침일(시간)' : 'time', '용도'  : 'gas_type', '비보정사용량' : 'raw_usage', '보정사용량': 'calib_usage',
                 '보정계수':'calib_const', '순간유량': 'flux'}).drop(0,axis=0)\
.assign(time=lambda x: pd.to_datetime(x['time']),
        year=lambda x: x['time'].dt.year.astype('str'),
        month=lambda x: x['time'].dt.month.astype('str'),
        hour=lambda x: x['time'].dt.hour.astype('str'),
        week= lambda x: x['time'].dt.weekday_name,
#        raw_usage=lambda x: x['raw_usage'].astype('float64'),
        calib_usage=lambda x: x['calib_usage'].astype('float'),
        calib_const=lambda x: x['calib_const'].astype('float'),
        flux=lambda x: x['flux'].astype('float')).drop(['납부자번호', '고객명', 'ID', '주소', 'gas_type', '설치번호', '설비번호'],axis=1)\
.query("time> '2018-01-25 00:00' and time<= '2020-01-16 00:00'").dropna()

# replace
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

apt_rawdata=apt_rawdata.drop(['calib_usage_1dayago','calib_usage_1weekago'], axis=1).set_index('time')


apt_rawdata.shape
apt_rawdata.index
apt_rawdata.columns
apt_rawdata.info()
apt_rawdata.count()
apt_rawdata.nunique() 
apt_rawdata.head().round(2)
apt_rawdata.isnull().sum()
(apt_rawdata.isnull().sum()/max(apt_rawdata.count())).round(2)

#numerical Value
# histogram
# columns name sort

apt_rawdata.describe().round(2).transpose()

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,12)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True 

plt.subplot(411)
plt.title("Hotel Usage(change over time)", fontsize=30)
plt.plot(apt_rawdata['time'], apt_rawdata['calib_usage'], linestyle='-', linewidth=2, color='r') # 'dashed'
plt.legend(['calib_usage'], fontsize=15, loc='best')
plt.ylabel("Usage", fontsize=15)
plt.subplot(412)
plt.plot(apt_rawdata['time'], apt_rawdata['raw_usage'], linestyle='-', linewidth=2, color='g') # 'dashed'
plt.legend(['raw_usage'], fontsize=15, loc='best')
plt.ylabel("raw_Usage", fontsize=15)
plt.subplot(413)
plt.plot(apt_rawdata['time'], apt_rawdata['calib_const'], linestyle='-', linewidth=2, color='b') # 'dashed'
plt.legend(['calib_const'], fontsize=15, loc='best')
plt.ylabel("calib_const", fontsize=15)
plt.subplot(414)
plt.plot(apt_rawdata['time'], apt_rawdata['flux'], linestyle='-', linewidth=2, color='black') # 'dashed'
plt.legend(['flux'], fontsize=15, loc='best')
plt.xlabel("Time(hour)", fontsize=15)
plt.ylabel("flux", fontsize=15)
plt.show()

import seaborn as sns
categorical_feature = list( set([ col for col in apt_rawdata.columns if apt_rawdata[col].dtypes == "object"]))
numerical_feature = list( set([ col for col in apt_rawdata.columns if apt_rawdata[col].dtypes in(['float64', 'int64']) ])-set(['time']))
time_feature = list(set(apt_rawdata.columns) - set(categorical_feature)-set(numerical_feature))

for col in numerical_feature:
    plt.rcParams["figure.figsize"] = (15,5)
    sns.distplot(apt_rawdata.loc[apt_rawdata[col].notnull(), col])
    plt.title(col)
    plt.show()

plt.subplot(411)
plt.title("APT Usage", fontsize=30)
apt_rawdata.calib_usage.plot.hist()
apt_rawdata.calib_usage.plot.kde()

# Categorical Feature
for col in categorical_feature:
    unique_list = apt_rawdata[col].unique()
    print(unique_list)

for col in categorical_feature:
#    apt_rawdata[col].value_counts().plot(kind='bar')
    apt_rawdata[col].value_counts().plot.pie(autopct='%.2f%%')
    print(apt_rawdata.groupby(col).mean())
    plt.title(col)
    plt.show()

apt_rawdata.month.value_counts().plot.pie(autopct='%.2f%%')

df=apt_rawdata.groupby(['year','month']).mean()
df=df.assign(year= df.index.droplevel(1),
             month= df.index.droplevel(0)).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='calib_usage', data=df, ax=axes[0])
sns.boxplot(x='month', y='calib_usage', data=df.loc[~df.year.isin([2018, 2020]), :])


# lag Plots
apt_rawdata.calib_usage=np.where(apt_rawdata.calib_usage>100,None,apt_rawdata.calib_usage)
from pandas.plotting import lag_plot
plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':8})
fig, axes = plt.subplots(8, 3, figsize=(15,15), sharex=True, sharey=True, dpi=110)
plt.subplots_adjust( hspace=0.5)
for i, ax in enumerate(axes.flatten()[:24]):
    lag_plot(apt_rawdata.calib_usage, lag=i+1, ax=ax, c='firebrick', alpha=0.5, s=3) #s:size, c:color, alpha:
    ax.set_title('Lag ' + str(i+1), fontsize=10)

fig.suptitle('Lag Plot', y=2)    


