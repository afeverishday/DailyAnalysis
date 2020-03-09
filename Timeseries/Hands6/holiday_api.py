#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:54:38 2020

@author: keti
"""


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

