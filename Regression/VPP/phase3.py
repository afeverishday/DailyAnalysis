#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:52:16 2020

@author: keti
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:48:04 2019

@author: keti
"""

import os
print(os.getcwd())

#MongoDB 연동
import pymongo
import pandas as pd
import gc
import numpy as np
import datetime
import sys
from sklearn import datasets
import pickle
from pytictoc import TicToc
from sklearn.externals import joblib
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')

crtntime = "202001201500"
# time= pd.to_datetime(time).strftime('%Y%m%d%H%M')
# time = sys.argv[1]
def phase3_QGEN(crtntime):
    print("CRTN_TM : {}".format(crtntime))
    connection =pymongo.MongoClient('10.0.1.40', 27017)
    db= connection.get_database('kma')
    nwp = db.get_collection('keti_nwp')
    tictoc = TicToc() #create instance of class
    
    tictoc.tic() #Start timer
    keti_nwp = pd.DataFrame(list(nwp.find( 
            { "CRTN_TM": pd.to_datetime(crtntime)},
            {"_id":False,"COMPX_ID":True,"CRTN_TM" : True, "LEAD_HR":True, 
             "NDNSW":True,"HFSFC":True, "TMP":True, "RH":True,'TMP-SFC':True})))\
                                       .drop_duplicates(['COMPX_ID',"LEAD_HR"], keep='last')\
                                       .sort_values(['COMPX_ID',"LEAD_HR"], ascending=False)
                                       #.query('CRTN_TM.dt.hour==10')

    tictoc.toc("read_nwp") #Time elapsed since t.tic()
    keti_nwp.isnull().sum()
        
    db= connection.get_database('sites')
    fcst_precision = db.get_collection('precision_new')
    tictoc.tic() #Start timer
    precision = pd.DataFrame(list(fcst_precision.find( 
            {"CRTN_TM": pd.to_datetime(crtntime)+ datetime.timedelta(hours=-3)},
            {"_id":False,"COMPX_ID":True,"CRTN_TM" : True,"fcst_tm_hour":True, "LEAD_HR":True, 
             "QGEN":True })))\
                                       .drop_duplicates(['COMPX_ID',"LEAD_HR"], keep='last')\
                                       .sort_values(['COMPX_ID' ,"LEAD_HR"], ascending=False)
    tictoc.toc("read_precision") #Time elapsed since t.tic()
    precision.isnull().sum()
    
    tictoc.tic() #Start timer
    fcst_prediction = db.get_collection('prediction_new')
    prediction = pd.DataFrame(list(fcst_prediction.find( 
            {"CRTN_TM": pd.to_datetime(crtntime) },
            {"_id":False,"COMPX_ID":True,"CRTN_TM" : True, "fcst_tm_hour":True, "LEAD_HR":True,
             "QGEN_echo":True,"QGEN_sejong1":True, "QGEN_sejong2":True,"QGEN_keti":True,
             "FCST_SRAD_SLOPE":True, "FCST_SRAD_HORIZ":True,"FCST_TEMP" :True, "FCST_MTEMP":True,
             "VOLUME":True, "SRAD_group":True})))\
                                       .drop_duplicates(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], keep='last')\
                                       .sort_values(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], ascending=False)
    
    tictoc.toc("read_prediction") #Time elapsed since t.tic()
    
    rawdata= pd.merge(precision, prediction, how="inner", on=['COMPX_ID',"CRTN_TM","LEAD_HR","fcst_tm_hour"])\
                 .merge(keti_nwp, how="inner", on=['COMPX_ID',"CRTN_TM","LEAD_HR"]).reset_index(drop=True)\
                 .round({'QGEN_echo':2, 'QGEN_sejong1':2,'QGEN_sejong2':2,'QGEN_keti':2})\
                 .drop_duplicates(['COMPX_ID',"CRTN_TM","LEAD_HR","fcst_tm_hour"], keep='last')\
                 .sort_values(['COMPX_ID','LEAD_HR','SRAD_group' ], ascending=False)\
                 .dropna(axis=0)\
                 .query('COMPX_ID not in ["E02M2127","P31S51030","P31S51031","P31S51032","P31S51033","P61M30540","P31S51034",\
                                          "P31S51035", "P61S30510", "P61S30520","P61S30530","P61S30610","P61S31180","P61S31270"]')\
                 .reset_index(drop=True)\
                 .assign(COMPX_ID=lambda x: x['COMPX_ID'].astype('object'))
#                 .query('(5<fcst_tm_hour)&(fcst_tm_hour<21)')\

#                 .drop(['SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour'], axis=1)



from pytictoc import TicToc
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import gc
import numpy as np
#working directory
import os
os.getcwd()
from sklearn import datasets
from sklearn.externals import joblib



    #multi model load
    tictoc.tic() #Time elapsed since t.tic()
    model_list= joblib.load('/home/keti/vpp/vpp_model/model_list_COMPX_ID.pkl')
    minmax_feature, target_feature, standard_feature, dummy_feature= joblib.load('/home/keti/vpp/vpp_model/feature_list_COMPX_ID.pkl')
    #multi model load
    model= joblib.load('/home/keti/vpp/vpp_model/model_COMPX_ID.pkl')
    
    tictoc.tic() #Time elapsed since t.tic()
    for i in np.arange(0,len(model_list),1):
        print(i)
        globals()[model_list[i]]= model[i]
    tictoc.toc("read_model") #Time elapsed since t.tic()


