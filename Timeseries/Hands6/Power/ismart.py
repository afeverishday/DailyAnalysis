#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:17:08 2020

@author: keti
"""

import gc
gc.collect()
import sys, os
os.getcwd()
import sys 
from os import rename, listdir 
import pandas as pd
import natsort 

# 현재 위치의 파일 목록 
directory='/home/keti/DataAnalysis/Heat_Analysis/keti/iSmart_201912'
files = listdir(directory+'/')

import natsort 
files=natsort.natsorted(files,reverse=False)

# 파일명에 번호 추가하기
for name in files: 
    # 파이썬 실행파일명은 변경하지 않음 
    if sys.argv[0].split("\\")[-1] == name: 
       continue 
    new_name = name.replace(".xls",".html" ) 
    os.rename(directory+'/'+name,directory+'/'+new_name)


i=0
df=pd.DataFrame()

for name in files:
    i=i+1
    df1=pd.read_html(directory+'/'+name )[7]
    df1.columns=['time','usage', 'peak','Reactive_1','Reactive_2','Active_1','Active_2']
    df1.time= pd.date_range(pd.to_datetime('2019-'+directory[-2:]+'-'+str(i)), pd.to_datetime('2019-'+directory[-2:]+'-'+str(i)+' 23:00'), freq='H')
    df=pd.concat([df,df1])

df.to_csv(directory+'.csv')


# total 현재 위치의 파일 목록 
directory='/home/keti/DataAnalysis/Heat_Analysis/keti/month'
files =listdir(directory)


files=natsort.natsorted(files,reverse=False)
df=pd.DataFrame()

for name in files:
    df1=pd.read_csv(directory+'/'+name )
    df1=df1.drop(['Unnamed: 0'],axis=1)
    df=pd.concat([df,df1])

df.to_csv(directory[:-5]+'hands6_power.csv')


