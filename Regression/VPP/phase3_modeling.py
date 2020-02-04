#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:15:00 2019

@author: keti
"""
#MongoDB 연동
import sys
import pymongo
import pandas as pd
import gc
import numpy as np
import datetime
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib.pyplot as plt

from pytictoc import TicToc
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')

gc.collect()

crtntime = "202001192300"


def phase3_model(crtntime):
    print("CRTN_TM : {}".format(crtntime))
    connection =pymongo.MongoClient('10.0.1.40', 27017)

    #collection list 
    #collection_list=db.collection_names()

    end_date = pd.to_datetime(crtntime)
    start_date= pd.to_datetime("201911080100")

    tictoc = TicToc() #create instance of class
    db= connection.get_database('kma')
    nwp = db.get_collection('keti_nwp')
    tictoc.tic() #Start timer
    keti_nwp = pd.DataFrame(list(nwp.find( 
            {"$and": 
                [ {"CRTN_TM": {"$lte":pd.to_datetime(end_date), "$gte":pd.to_datetime(start_date)}},
#                  {"fcst_tm_hour":{"$gte":8,"$lte":18}},
                  ]},
            {"_id":False,"COMPX_ID":True,"CRTN_TM" : True, "LEAD_HR":True, "NDNSW":True,"HFSFC":True, "TMP":True,
             "RH":True,'TMP-SFC':True})))\
                                       .drop_duplicates(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], keep='last')\
                                       .sort_values(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], ascending=False)\
                                       .query('CRTN_TM.dt.hour==10')

    tictoc.toc("read_nwp") #Time elapsed since t.tic()
    keti_nwp.isnull().sum()
    
    
    db= connection.get_database('sites')
    fcst_precision = db.get_collection('precision_new')
    tictoc.tic() #Start timer
    precision = pd.DataFrame(list(fcst_precision.find( 
            {"$and": 
                [ {"CRTN_TM": {"$lte":pd.to_datetime(end_date), "$gte":pd.to_datetime(start_date)}},
#                  {"fcst_tm_hour":{"$gte":8,"$lte":18}},
                  {"crtm_tm_hour":10}]},
            {"_id":False,"COMPX_ID":True,"CRTN_TM" : True,"fcst_tm_hour":True, "LEAD_HR":True, 
             "QGEN":True})))\
                                       .drop_duplicates(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], keep='last')\
                                       .sort_values(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], ascending=False)
    tictoc.toc("read_precision") #Time elapsed since t.tic()
    precision.isnull().sum()
    
    tictoc.tic() #Start timer
    fcst_prediction = db.get_collection('prediction_new')
    prediction = pd.DataFrame(list(fcst_prediction.find( 
            {"$and": 
                [{"CRTN_TM": {"$lte":pd.to_datetime(end_date), "$gte":pd.to_datetime(start_date)} },
#                 {"fcst_tm_hour":{"$gte":8,"$lte":18}},
                 {"crtm_tm_hour":10}]},
            {"_id":False,"COMPX_ID":True,"CRTN_TM" : True, "fcst_tm_hour":True, "LEAD_HR":True,
             "QGEN_echo":True,"QGEN_sejong1":True, "QGEN_sejong2":True,"QGEN_keti":True,
             "FCST_SRAD_SLOPE":True, "FCST_SRAD_HORIZ":True,"FCST_TEMP" :True, "FCST_MTEMP":True,
             "VOLUME":True, "SRAD_group":True })))\
                                       .drop_duplicates(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], keep='last')\
                                       .sort_values(['COMPX_ID',"CRTN_TM" ,"LEAD_HR"], ascending=False)
    
    tictoc.toc("read_prediction") #Time elapsed since t.tic()
    
    rawdata= pd.merge(precision, prediction, how="inner", on=['COMPX_ID',"CRTN_TM","LEAD_HR","fcst_tm_hour"])\
                 .merge(keti_nwp, how="inner", on=['COMPX_ID',"CRTN_TM","LEAD_HR"]).reset_index(drop=True)\
                 .round({'QGEN_echo':2, 'QGEN_sejong1':2,'QGEN_sejong2':2,'QGEN_keti':2})\
                 .drop_duplicates(['COMPX_ID',"CRTN_TM","LEAD_HR","fcst_tm_hour"], keep='last')\
                 .sort_values(['COMPX_ID','LEAD_HR','SRAD_group' ], ascending=False)\
                 .dropna(axis=0)\
                 .query('(5<fcst_tm_hour)&(fcst_tm_hour<21)')\
                 .query('COMPX_ID not in ["E02M2127","P31S51030","P31S51031","P31S51032","P31S51033","P61M30540","P31S51034",\
                                      "P31S51035", "P61S30510", "P61S30520","P61S30530","P61S30610","P61S31180","P61S31270"]')\
                 .reset_index(drop=True)\
                 .assign(COMPX_ID=lambda x: x['COMPX_ID'].astype('object'))
#                 .drop(['SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour'], axis=1)

    len(rawdata.COMPX_ID.unique())

    rawdata.columns
    rawdata.head().round(2)
    rawdata.describe().round(2)
    rawdata.info()
    (rawdata.isnull().sum()/max(rawdata.count())).round(2)
        
    # columns name sort
    categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"])-set(['QGEN']))
    numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ])-set(['COMPX_ID','LEAD_HR','CRTN_TM','fcst_tm_hour']))
    time_feature= list(set(rawdata.columns) - set(categorical_feature)-set(numerical_feature)-set(['COMPX_ID']))
    
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['QGEN'])],\
                                                         rawdata['QGEN'], test_size=0.1, random_state=777, stratify=rawdata['COMPX_ID'])
    import collections
    collections.Counter(rawdata['COMPX_ID'])  
    collections.Counter(train_x['COMPX_ID'])
    collections.Counter(test_x['COMPX_ID'])
    

    from sklearn.metrics import mutual_info_score
    from minepy import MINE
    #correlation
    corr_tot=rawdata.drop(['COMPX_ID','LEAD_HR','VOLUME','fcst_tm_hour' ], axis=1).corr(method='pearson').round(2)
    
    mine = MINE()
    MINE().compute_score(rawdata['QGEN_echo'], rawdata['QGEN'])
    MINE().compute_score(rawdata['QGEN_sejong1'], rawdata['QGEN'])
    MINE().compute_score(rawdata['QGEN_sejong2'], rawdata['QGEN'])
    MINE().compute_score(rawdata['QGEN_keti'], rawdata['QGEN'])
    
    
    from pandas.plotting import scatter_matrix
    scatter_matrix(rawdata, diagonal='kde', color='b', alpha=0.3, figsize=(20, 15))
    plt.scatter(rawdata['QGEN_echo'], rawdata['QGEN'])
    plt.title("MIC={0:0.3f}".format(mine.mic()))
    
    plt.figure(figsize=(10,10))
    sns.heatmap(data = df.corr(method='pearson'),
                annot=True,fmt = '.2f', linewidths=.2, cmap='Blues', square=True)
    
          
    # modeling
    linear_model= LinearRegression(fit_intercept=True).fit(train_x, train_y)
    ridge_model= Ridge().fit(train_x, train_y)
    lasso_model= Lasso().fit(train_x, train_y)
    tree_model= DecisionTreeRegressor().fit(train_x, train_y)  
    xgb_model= xgboost.XGBRegressor().fit(train_x, train_y)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    len(phase3_model.COMPX_ID.unique())
    #correlation
    # COMPX_ID model==================================================================================
    #a=['P31S2105']
    #phase3_model= phase3_model[phase3_model['COMPX_ID'].isin(a)].reset_index(drop=True)
 
    phase3_model.dtypes
    phase3_model.columns
    from sklearn.model_selection import train_test_split

       # 컬럼만 있는 데이터프레임 생성
    corr_COMPX = pd.DataFrame(columns=['QGEN_echo','QGEN_sejong1','QGEN_sejong2','QGEN_keti','FCST_SRAD_SLOPE','FCST_SRAD_HORIZ', 'FCST_TEMP','FCST_MTEMP',
                                       'NDNSW', 'HFSFC', 'TMP', 'RH', 'TMP-SFC', 'COMPX_ID'])
    phase3_COMPX = pd.DataFrame(columns=['QGEN','QGEN_echo','QGEN_sejong1','QGEN_sejong2','QGEN_keti','COMPX_ID','VOLUME', 
                                         'pred_linear','pred_lasso', 'pred_ridge','pred_tree','pred_xgb'])
    
    data=phase3_model
    # data split
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.svm import LinearSVR, SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    import xgboost
    from xgboost import plot_importance
    import sklearn

    print('sklearn version: %s' % sklearn.__version__)
    print('xgboost version: %s' % xgboost.__version__)
    from sklearn.metrics import explained_variance_score
 
#   model_list=[]
    #model validate
    for COMPX_ID,data in phase3_model.groupby('COMPX_ID'):
        # Ridge,Lasso Regression
        print(COMPX_ID)
        print('predict_model_'+COMPX_ID)
        corr=data.drop(['LEAD_HR','VOLUME','fcst_tm_hour' ], axis=1).corr(method='pearson')\
        .assign(COMPX_ID=COMPX_ID).reset_index(drop=True).drop(["QGEN"],axis=1).head(1).round(2)
        
        train, test = train_test_split(data, test_size= 0.3, random_state=1234)
        train_x=train.drop([ 'QGEN','COMPX_ID','SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour','VOLUME'],axis=1)
        train_y=train[['QGEN']]
        test_x=test.drop([ 'QGEN','COMPX_ID','SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour','VOLUME'],axis=1)
        test_y=test[['QGEN']]
        globals()['linear_model_{}'.format(COMPX_ID)] = LinearRegression(fit_intercept=True).fit(train_x, train_y)
        globals()['tree_model_{}'.format(COMPX_ID)] = DecisionTreeRegressor().fit(train_x, train_y) 
        globals()['xgb_model_{}'.format(COMPX_ID)] = xgboost.XGBRegressor(objective ='reg:squarederror').fit(train_x, train_y)

#        ridge_model= Ridge().fit(train_x, train_y)
#        lasso_model= Lasso().fit(train_x, train_y)
#        linsvr_model= LinearSVR().fit(train_x, train_y)

        #model save
#        joblib.dump(globals()['linear_model_{}'.format(COMPX_ID)],'/home/keti/vpp/vpp_model/linear_model'+COMPX_ID+'.pkl' )
#        joblib.dump(ridge_model,'/home/keti/vpp/vpp_model/ridge_model'+COMPX_ID+'.pkl' )
#        joblib.dump(lasso_model,'/home/keti/vpp/vpp_model/lasso_model'+COMPX_ID+'.pkl' )
#        joblib.dump(linsvr_model,'/home/keti/vpp/vpp_model/linsvr_model'+COMPX_ID+'.pkl' )
#        joblib.dump(globals()['tree_model_{}'.format(COMPX_ID)],'/home/keti/vpp/vpp_model/tree_model'+COMPX_ID+'.pkl' )
#        joblib.dump(globals()['xgb_model_{}'.format(COMPX_ID)],'/home/keti/vpp/vpp_model/xgb_model'+COMPX_ID+'.pkl' )
        
        result= test[['QGEN','QGEN_echo','QGEN_sejong1','QGEN_sejong2','QGEN_keti','COMPX_ID','VOLUME']]\
            .assign(pred_linear= np.where(globals()['linear_model_{}'.format(COMPX_ID)] .predict(test_x)<0,0, 
                                      np.where(globals()['linear_model_{}'.format(COMPX_ID)] .predict(test_x)>np.array(test.VOLUME)[1],
                                               np.array(test.VOLUME)[1] ,globals()['linear_model_{}'.format(COMPX_ID)].predict(test_x))),
#                   pred_lasso= np.where(lasso_model.predict(test_x)<0,0, 
#                                      np.where(lasso_model.predict(test_x)>np.array(test.VOLUME)[1],np.array(test.VOLUME)[1] ,lasso_model.predict(test_x)) ),
#                   pred_ridge= np.where(ridge_model.predict(test_x)<0,0, 
#                                      np.where(ridge_model.predict(test_x)>np.array(test.VOLUME)[1],np.array(test.VOLUME)[1] ,ridge_model.predict(test_x)) ),
#                   pred_linsvr= np.where(linsvr_model.predict(test_x)<0,0, 
#                                      np.where(linsvr_model.predict(test_x)>np.array(test.VOLUME)[1],np.array(test.VOLUME)[1] ,linsvr_model.predict(test_x)) ),
                    pred_tree= np.where(globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x)<0,0, 
                                      np.where(globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x)>np.array(test.VOLUME)[1],
                                               np.array(test.VOLUME)[1] ,globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x)) ),
                    pred_xgb= np.where(globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x)<0,0, 
                                      np.where(globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x)>np.array(test.VOLUME)[1],
                                               np.array(test.VOLUME)[1] ,globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x)) ))               
        corr_COMPX= pd.concat([corr_COMPX, corr], axis=0)
        phase3_COMPX= pd.concat([phase3_COMPX, result], axis=0)
    
# echo precision 0<0.6 
#  ['P31S51020','P31S51021','P31S51022','P31S51023','P31S51024','P31S51025','P31S51026','P31S51027','P31S51028','P31S51029','P31S51036','P31S51037','P31S51038',
#   'P31S51039','P31S5103A','P31S5103B','P31S51073','P32S90001','P61M31220','P61S2101','P61S2102','P61S2103','P61S21520','P61S30560','P61S30590','P61S30600',
#   'P61S31610', 'P63S2112', 'P63S32010', 'P63S32030', 'P64M52010','P64M52030','P64M52050','P64M52130','P64S52080','P64S91003']

    #list(corr_COMPX.query('QGEN_echo<0.6').COMPX_ID)
    del(train, test, train_x, train_y, test_x, test_y, result)

    phase3_COMPX = phase3_COMPX.assign(nmape_echo= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME,(abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_echo)/phase3_COMPX.VOLUME)*100, None),
                           nmape_sejong1= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1,(abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_sejong1)/phase3_COMPX.VOLUME)*100, None),
                           nmape_sejong2= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1,(abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_sejong2)/phase3_COMPX.VOLUME)*100, None),
                           nmape_keti= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1,(abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_keti)/phase3_COMPX.VOLUME)*100, None),
                           nmape_linear= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1,(abs(phase3_COMPX.QGEN-phase3_COMPX.pred_linear)/phase3_COMPX.VOLUME)*100, None),
#                           nmape_ridge= (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_ridge)/phase3_COMPX.VOLUME)*100,
#                           nmape_lasso= (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_lasso)/phase3_COMPX.VOLUME)*100,
#                           nmape_linsvr= (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_linsvr)/phase3_COMPX.VOLUME)*100,
                           nmape_tree= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1,(abs(phase3_COMPX.QGEN-phase3_COMPX.pred_tree)/phase3_COMPX.VOLUME)*100, None),
                           nmape_xgb= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1,(abs(phase3_COMPX.QGEN-phase3_COMPX.pred_xgb)/phase3_COMPX.VOLUME)*100, None),
                           mape_echo= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_echo)/phase3_COMPX.QGEN)*100, None),
                           mape_sejong1= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_sejong1)/phase3_COMPX.QGEN)*100, None),
                           mape_sejong2= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_sejong2)/phase3_COMPX.QGEN)*100, None),
                           mape_keti= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.QGEN_keti)/phase3_COMPX.QGEN)*100, None),
                           mape_linear= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_linear)/phase3_COMPX.QGEN)*100, None),
#                           mape_ridge= np.where(phase3_COMPX.QGEN > 1, (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_ridge)/phase3_COMPX.QGEN)*100, None),
#                           mape_lasso= np.where(phase3_COMPX.QGEN > 1, (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_lasso)/phase3_COMPX.QGEN)*100, None),
#                           mape_linsvr= np.where(phase3_COMPX.QGEN > 1, (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_linsvr)/phase3_COMPX.QGEN)*100, None),
                           mape_tree= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_tree)/phase3_COMPX.QGEN)*100, None),
                           mape_xgb= np.where(phase3_COMPX.QGEN > phase3_COMPX.VOLUME*0.1, (abs(phase3_COMPX.QGEN-phase3_COMPX.pred_xgb)/phase3_COMPX.QGEN)*100, None))
   
    print( "nmape_echo:{},nmape_echo_max:{},mape_echo_mean:{},mape_echo_max:{}".format(phase3_COMPX.nmape_echo.mean(axis=0), phase3_COMPX.nmape_echo.max(axis=0),phase3_COMPX.mape_echo.mean(axis=0), phase3_COMPX.mape_echo.max(axis=0)))
    print( "nmape_sejong1:{},nmape_sejong1_max:{},mape_sejong1_mean:{},mape_sejong1_max:{}".format(phase3_COMPX.nmape_sejong1.mean(axis=0), phase3_COMPX.nmape_sejong1.max(axis=0),phase3_COMPX.mape_sejong1.mean(axis=0), phase3_COMPX.mape_sejong1.max(axis=0)))
    print( "nmape_sejong2:{},nmape_sejong2_max:{},mape_sejong2_mean:{},mape_sejong2_max:{}".format(phase3_COMPX.nmape_sejong2.mean(axis=0), phase3_COMPX.nmape_sejong2.max(axis=0),phase3_COMPX.mape_sejong2.mean(axis=0), phase3_COMPX.mape_sejong2.max(axis=0)))
    print( "nmape_keti:{},nmape_keti_max:{},mape_keti_mean:{},mape_keti_max:{}".format(phase3_COMPX.nmape_keti.mean(axis=0), phase3_COMPX.nmape_keti.max(axis=0),phase3_COMPX.mape_keti.mean(axis=0), phase3_COMPX.mape_keti.max(axis=0)))
    print( "nmape_linear:{},nmape_linear_max:{},mape_linear_mean:{},mape_linear_max:{}".format(phase3_COMPX.nmape_linear.mean(axis=0), phase3_COMPX.nmape_linear.max(axis=0),phase3_COMPX.mape_linear.mean(axis=0), phase3_COMPX.mape_linear.max(axis=0)))
#    print( "nmape_lasso:{},nmape_lasso_max:{},mape_lasso_mean:{},mape_lasso_max:{}".format(phase3_COMPX.nmape_lasso.mean(axis=0), phase3_COMPX.nmape_lasso.max(axis=0),phase3_COMPX.mape_lasso.mean(axis=0), phase3_COMPX.mape_lasso.max(axis=0)))
#    print( "nmape_ridge:{},nmape_ridge_max:{},mape_ridge_mean:{},mape_ridge_max:{}".format(phase3_COMPX.nmape_ridge.mean(axis=0), phase3_COMPX.nmape_ridge.max(axis=0),phase3_COMPX.mape_ridge.mean(axis=0), phase3_COMPX.mape_ridge.max(axis=0)))
#    print( "nmape_linsvr:{},nmape_linsvr_max:{},mape_linsvr_mean:{},mape_linsvr_max:{}".format(phase3_COMPX.nmape_linsvr.mean(axis=0), phase3_COMPX.nmape_linsvr.max(axis=0),phase3_COMPX.mape_linsvr.mean(axis=0), phase3_COMPX.mape_linsvr.max(axis=0)))
    print( "nmape_tree:{},nmape_tree_max:{},mape_tree_mean:{},mape_tree_max:{}".format(phase3_COMPX.nmape_tree.mean(axis=0), phase3_COMPX.nmape_tree.max(axis=0),phase3_COMPX.mape_tree.mean(axis=0), phase3_COMPX.mape_tree.max(axis=0)))
    print( "nmape_xgb:{},nmape_xgb_max:{},mape_xgb_mean:{},mape_xgb_max:{}".format(phase3_COMPX.nmape_xgb.mean(axis=0), phase3_COMPX.nmape_xgb.max(axis=0),phase3_COMPX.mape_xgb.mean(axis=0), phase3_COMPX.mape_xgb.max(axis=0)))

   


 
 

# total model==================================================================================
    import os
    print (os.getcwd()) #현재 디렉토리의
 
    #list(corr_COMPX.query('QGEN_echo<0.8').COMPX_ID)
    # echo precision 0<0.6 
#  ['P31S51020','P31S51021','P31S51022','P31S51023','P31S51024','P31S51025','P31S51026','P31S51027','P31S51028','P31S51029','P31S51036','P31S51037','P31S51038',
#   'P31S51039','P31S5103A','P31S5103B','P31S51073','P32S90001','P61M31220','P61S2101','P61S2102','P61S2103','P61S21520','P61S30560','P61S30590','P61S30600',
#   'P61S31610', 'P63S2112', 'P63S32010', 'P63S32030', 'P64M52010','P64M52030','P64M52050','P64M52130','P64S52080','P64S91003']

    phase3_model=phase3_model[phase3_model['COMPX_ID'].isin(["P61S30560","P61S2101","P61S2103", "P64M52030","P613S31610","P61S30600","P32S90001",
                              "P64M52010","P63S32010","P61S21520",'P31S51020', 'P31S51021', 'P31S51022', 'P31S51023', 'P31S51024',
                              'P31S51025', 'P31S51026', 'P31S51027', 'P31S51028', 'P31S51029', 'P31S51036', 'P31S51037', 'P31S51038',
                              'P31S51039', 'P31S5103A', 'P31S5103B', 'P31S51040', 'P31S51041', 'P31S51054', 'P31S51073', 'P31S51086',
                              'P31S51093', 'P31S51130', 'P31S51131', 'P31S51132', 'P31S51133', 'P31S51134', 'P31S51136', 'P31S51137',
                              'P31S51140', 'P31S51150', 'P31S51151', 'P31S51152', 'P31S51153', 'P31S51154', 'P31S51155', 'P31S51156',
                              'P31S51157', 'P32S90001', 'P41S2124', 'P41S21470', 'P41S21473', 'P41S21475', 'P41S21500', 'P41S21501',
                              'P41S21503', 'P43S5110A', 'P54M51010', 'P54M51011', 'P61L31020', 'P61M31050', 'P61M31220', 'P61M31250',
                              'P61M31560', 'P61M61120', 'P61S2101', 'P61S2102', 'P61S2103', 'P61S21520', 'P61S30550', 'P61S30560',
                              'P61S30570', 'P61S30590', 'P61S30600', 'P61S30630', 'P61S30660', 'P61S30670', 'P61S31130', 'P61S31290',
                              'P61S31450', 'P61S31451', 'P61S31452', 'P61S31453', 'P61S31610', 'P63M2106', 'P63M32100', 'P63S2110',
                              'P63S2112', 'P63S2114', 'P63S2115', 'P63S2202', 'P63S32010', 'P63S32030', 'P63S32080', 'P63S32120',
                              'P64M52010', 'P64M52020', 'P64M52030', 'P64M52040', 'P64M52050', 'P64M52070', 'P64M52090', 'P64M52100',
                              'P64M52130', 'P64S52060', 'P64S52080', 'P64S52110', 'P64S52120', 'P64S91003' ])==False].reset_index(drop=True)
    
    phase3_model=phase3_model[phase3_model['COMPX_ID'].isin(["P61S30560","P61S2101","P61S2103", "P64M52030",
                              "P613S31610","P61S30600","P32S90001","P64M52010","P63S32010","P61S21520" ])==False].reset_index(drop=True)
    
    #phase3_model = phase3_model[phase3_model.COMPX_ID.str[0]!='W']
    #phase3_model = phase3_model[phase3_model.COMPX_ID=="P61S30590"]
    phase3_model.dtypes
    phase3_model.head(1)

    #model validate
    from sklearn.model_selection import train_test_split

    corr=phase3_model.drop(['COMPX_ID','LEAD_HR','VOLUME','fcst_tm_hour' ], axis=1).corr(method='pearson')\
        .reset_index(drop=True).drop(["QGEN"],axis=1).head(1).round(2)
    train, test = train_test_split(phase3_model,test_size= 0.3, random_state=1234)
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
    train_x=train.drop([ 'QGEN','COMPX_ID','SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour'],axis=1)
    train_y=train[['QGEN']]
    test_x=test.drop([ 'QGEN','COMPX_ID','SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour'],axis=1)
    test_y=test[['QGEN']]
    linear_model= LinearRegression(fit_intercept=True).fit(train_x, train_y)
    ridge_model= Ridge().fit(train_x, train_y)
    lasso_model= Lasso().fit(train_x, train_y)
    tree_model= DecisionTreeRegressor().fit(train_x, train_y)  
    xgb_model= xgboost.XGBRegressor().fit(train_x, train_y)  

    phase3= test[['QGEN','QGEN_echo','QGEN_sejong1', 'QGEN_sejong2','QGEN_keti','COMPX_ID','VOLUME']]\
            .assign(pred_linear= np.where(linear_model.predict(test_x)<0,0, linear_model.predict(test_x)),
                    pred_lasso= np.where(lasso_model.predict(test_x)<0,0,lasso_model.predict(test_x)),
                    pred_ridge= np.where(ridge_model.predict(test_x)<0,0,ridge_model.predict(test_x)),
                    pred_tree= np.where(tree_model.predict(test_x)<0,0,tree_model.predict(test_x)),
                    pred_xgb= np.where(xgb_model.predict(test_x)<0,0,xgb_model.predict(test_x)) )   
    phase3= phase3.assign(pred_linear= np.where(phase3.pred_linear>phase3.VOLUME,phase3.VOLUME,phase3.pred_linear ),
                          pred_lasso= np.where(phase3.pred_lasso>phase3.VOLUME,phase3.VOLUME,phase3.pred_lasso ),
                          pred_ridge= np.where(phase3.pred_ridge>phase3.VOLUME,phase3.VOLUME,phase3.pred_ridge ),
                          pred_tree= np.where(phase3.pred_tree>phase3.VOLUME,phase3.VOLUME,phase3.pred_tree ),
                          pred_xgb= np.where(phase3.pred_xgb>phase3.VOLUME,phase3.VOLUME,phase3.pred_xgb ))      
                    
    phase3 = phase3.assign(nmape_echo= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_echo)/phase3.VOLUME)*100, None),
                           nmape_sejong1= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_sejong1)/phase3.VOLUME)*100, None),
                           nmape_sejong2= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_sejong2)/phase3.VOLUME)*100, None),
                           nmape_keti= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_keti)/phase3.VOLUME)*100, None),
                           nmape_linear= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_linear)/phase3.VOLUME)*100, None),
                           nmape_ridge= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_ridge)/phase3.VOLUME)*100, None),
                           nmape_lasso= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_lasso)/phase3.VOLUME)*100, None),
                           nmape_tree= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_tree)/phase3.VOLUME)*100, None),
                           nmape_xgb= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_xgb)/phase3.VOLUME)*100, None),
                           mape_echo= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_echo)/phase3.QGEN)*100, None),
                           mape_sejong1= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_sejong1)/phase3.QGEN)*100, None),
                           mape_sejong2= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_sejong2)/phase3.QGEN)*100, None),
                           mape_keti= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.QGEN_keti)/phase3.QGEN)*100, None),                           
                           mape_linear= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_linear)/phase3.QGEN)*100, None),
                           mape_ridge= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_ridge)/phase3.QGEN)*100, None),
                           mape_lasso= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_lasso)/phase3.QGEN)*100, None),
                           mape_tree= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_tree)/phase3.QGEN)*100, None),
                           mape_xgb= np.where(phase3.QGEN > phase3.VOLUME*0.1, (abs(phase3.QGEN-phase3.pred_xgb)/phase3.QGEN)*100, None))

    print( "nmape_echo:{},nmape_echo_max:{},mape_echo:{},mape_echo_max:{}".format(phase3.nmape_echo.mean(axis=0), phase3.nmape_echo.max(axis=0),phase3.mape_echo.mean(axis=0), phase3.mape_echo.max(axis=0)))
    print( "nmape_sejong1:{},nmape_sejong1_max:{},mape_sejong1:{},mape_sejong1_max:{}".format(phase3.nmape_sejong1.mean(axis=0), phase3.nmape_sejong1.max(axis=0),phase3.mape_sejong1.mean(axis=0), phase3.mape_sejong1.max(axis=0)))
    print( "nmape_sejong2:{},nmape_sejong2_max:{},mape_sejong2:{},mape_sejong2_max:{}".format(phase3.nmape_sejong2.mean(axis=0), phase3.nmape_sejong2.max(axis=0),phase3.mape_sejong2.mean(axis=0), phase3.mape_sejong2.max(axis=0)))
    print( "nmape_keti:{},nmape_keti_max:{},mape_keti:{},mape_keti_max:{}".format(phase3.nmape_keti.mean(axis=0), phase3.nmape_keti.max(axis=0),phase3.mape_keti.mean(axis=0), phase3.mape_keti.max(axis=0)))
    print( "nmape_linear:{},nmape_linear_max:{},mape_linear:{},mape_linear_max:{}".format(phase3.nmape_linear.mean(axis=0), phase3.nmape_linear.max(axis=0),phase3.mape_linear.mean(axis=0), phase3.mape_linear.max(axis=0)))
    print( "nmape_lasso:{},nmape_lasso_max:{},mape_lasso:{},mape_lasso_max:{}".format(phase3.nmape_lasso.mean(axis=0), phase3.nmape_lasso.max(axis=0),phase3.mape_lasso.mean(axis=0), phase3.mape_lasso.max(axis=0)))
    print( "nmape_ridge:{},nmape_ridge_max:{},mape_ridge:{},mape_ridge_max:{}".format(phase3.nmape_ridge.mean(axis=0), phase3.nmape_ridge.max(axis=0),phase3.mape_ridge.mean(axis=0), phase3.mape_ridge.max(axis=0)))
    print( "nmape_tree:{},nmape_tree_max:{},mape_tree:{},mape_tree_max:{}".format(phase3.nmape_tree.mean(axis=0), phase3.nmape_tree.max(axis=0),phase3.mape_tree.mean(axis=0), phase3.mape_tree.max(axis=0)))
    print( "nmape_xgb:{},nmape_xgb_max:{},mape_xgb:{},mape_xgb_max:{}".format(phase3.nmape_xgb.mean(axis=0), phase3.nmape_xgb.max(axis=0),phase3.mape_xgb.mean(axis=0), phase3.mape_xgb.max(axis=0)))
  

    #model save
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    import xgboost

    corr_tot=phase3_model.drop(['COMPX_ID','LEAD_HR','VOLUME','fcst_tm_hour' ], axis=1).corr(method='pearson')
    train_x=phase3_model.drop([ 'QGEN','COMPX_ID','SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour'],axis=1)
    train_y=phase3_model[['QGEN']]
    linear_model= LinearRegression(fit_intercept=True).fit(train_x, train_y)
    ridge_model= Ridge().fit(train_x, train_y)
    lasso_model= Lasso().fit(train_x, train_y)
    tree_model= DecisionTreeRegressor().fit(train_x, train_y)  
    xgb_model= xgboost.XGBRegressor().fit(train_x, train_y)  

    from sklearn import datasets
    from sklearn.externals import joblib
    #multi model save
    joblib.dump([model_list],'/home/keti/vpp/vpp_model/model_test.pkl' )
    del(model_list)
    #multi model load
    linear_model,ridge_model,lasso_model,tree_model,xgb_model= joblib.load('/home/keti/vpp/vpp_model/model_test.pkl')

linear_model_P63S2202

























    
    
    # scale
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    df_train.dtypes    
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    df_train_minmax = min_max_scaler.fit(df_train)
    df_train_standard = standard_scaler.fit(df_train)
    df_test_minmax = min_max_scaler.fit(df_test)
    df_test_standard = standard_scaler.fit(df_test)
    print(df_train_minmax.data_max_)
    print(df_train_scale.data_min_)
    print(df_train_standard.var_)
    print(df_train_standard.mean_)
    
    df = min_max_scaler.transform(phase3_model)
    df_inverse= min_max_scaler.inverse_transform(df)
    df_inverse = pd.DataFrame(df_inverse, columns=phase3_model.columns, index=list(phase3_model.index.values)) 
    df = pd.DataFrame(df, columns=phase3_model.columns, index=list(phase3_model.index.values))
    print(df.head())
    
    
    from sklearn.metrics import mutual_info_score
    
    #correlation
    phase3_model.corr(method='pearson').round(2)
    df_train.corr(method='pearson').round(2)
    df_test.corr(method='pearson').round(2)

    from minepy import MINE
    mine = MINE()
    mine.compute_score(df_test['QGEN_echo'], y_test)
    mine.compute_score(df_test['QGEN_sejong1'], y_test)
    mine.compute_score(df_test['QGEN_sejong2'], y_test)
    mine.compute_score(df_test['QGEN_echo'], y_test)
    
    
    from pandas.plotting import scatter_matrix
    scatter_matrix(df_test, diagonal='kde', color='b', alpha=0.3, figsize=(20, 15))
    plt.scatter(df_test['QGEN_echo'], y_test)
    plt.title("MIC={0:0.3f}".format(mine.mic()))
    
    plt.figure(figsize=(10,10))
    sns.heatmap(data = df.corr(method='pearson'),
                annot=True,fmt = '.2f', linewidths=.2, cmap='Blues', square=True)
    
    # modeling
    from sklearn.model_selection import GridSearchCV

    from sklearn.linear_model import LinearRegression

    import statsmodels as sm
    import statsmodels.formula.api as smf
    # Linear Regression
    #model = LinearRegression(fit_intercept=True).fit(train, train.QGEN)
    #print(model.coef_, model.intercept_)
    linear_model = smf.ols('QGEN~ VOLUME +QGEN_echo +FCST_SRAD_SLOPE + FCST_TEMP +FCST_MTEMP+ QGEN_sejong2+QGEN_sejong1+FCST_SRAD_HORIZ', data=train).fit()
    print(linear_model.summary())
    pred_linear=linear_model.predict(test.drop(['QGEN'], axis=1))
    abs(test.QGEN-pred_linear).mean()
    
    # Ridge,Lasso Regression
    from sklearn.linear_model import Ridge, Lasso
    ridge_model= Ridge().fit(train.drop(['QGEN','COMPX_ID','CRTN_TM'], axis=1),train.QGEN)  
    lasso_model= Lasso().fit(train.drop(['QGEN','COMPX_ID','CRTN_TM'], axis=1),train.QGEN)  
    pred_ridge = ridge_model.predict(test.drop(['QGEN','COMPX_ID','CRTN_TM'], axis=1))
    pred_lasso = lasso_model.predict(test.drop(['QGEN','COMPX_ID','CRTN_TM'], axis=1))
    abs(test.QGEN-pred_linear).mean()
    abs(test.QGEN-pred_linear).mean()

    # svm Regression
    from sklearn.svm import LinearSVR, SVR
    linsvr_model= LinearSVR().fit(X_test, y_test)
    svr_model= SVR(kernel='poly').fit(X_test, y_test)
    pred_linsvr = linsvr_model.predict(X_test)
    pred_svr = svr_model.predict(X_test)
    abs(y_test-pred_linsvr).mean()
    abs(y_test-pred_svr).mean()

    
    # decisionTree Regression
    from sklearn.tree import DecisionTreeRegressor
    tree_model= DecisionTreeRegressor().fit(X_test, y_test)  
    pred_tree = tree_model.predict(X_test)
    abs(y_test-pred_tree).mean()
    
    # xgboost model
    import xgboost
    from xgboost import plot_importance
    
    xgb_model = xgboost.XGBRegressor().fit(X_test,y_test)
    pred_xgb = xgb_model.predict(X_test)
    abs(y_test-pred_xgb).mean()


    def get_xgb_imp(xgb, feat_names):
        from numpy import array
        imp_vals = xgb.booster().get_fscore()
        imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
        total = array(imp_dict.values()).sum()
        return {k:v/total for k,v in imp_dict.items()}


    # train_x와 train_y는 학습하고자 하는 트레이닝 셋의 설명변수와 종속변수의 ndarray 타입이다.
    model.fit(train_x, train_y) 
    feat_names = [i for i in range(0,100)] # 변수가 100개인 경우 가정 
    feature_importance = get_xgb_imp(model,feat_names)





    #model save
    from sklearn import datasets
    import pickle
    from sklearn.externals import joblib
    joblib.dump(xgb_model,'xgb_model_test.pkl' )
    #model load
    xgb_model = joblib.load('xgb_model_test.pkl')
    pred_xgb = xgb_model.predict(X_test)
    
    # result plot
    plt.semilogy(X_test['CRTN_TM'], pred_tree, label='tree predict', ls='--', dashes=(2,1))
    plt.semilogy(X_test['CRTN_TM'], pred_linear, label='linear predict', ls=':')
    plt.semilogy(data_train['date'], data_train['price'], label='train data', alpha=0.4)
    plt.semilogy(data_test['date'], data_test['price'], label='test data')
    plt.legend(loc=1)
    plt.xlabel('time', size=15)
    plt.ylabel('QGEN', size=15)
    plt.show()
    
    
    # xgboost model
    import xgboost
    from xgboost import plot_importance
        # 파라메터 후보
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

        # 그리드 서치 진행
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)        

        # 최종 모델 성능 점검
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23)

    grid_search.fit(X_train, y_train)
    grid_search.score(X_test, y_test)

    xgb_model = xgboost.XGBRegressor()
    
    
    xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 

    xgb_model.fit(X_test,y_test)
    from collections import OrderedDict
    from xgboost import plot_importance
    OrderedDict(sorted(xgb_model.booster().get_fscore().items(), key=lambda t: t[1], reverse=True))
    
    
    
    scale.inverse_transform(X_test)
    X_test['Prediction'] = np.expm1(xgb_model.predict(X_test))
    
    
    X_test['error'] = y_test -X_test['Prediction']
    
    sns.pairplot(phase3_model)
    plt.title("Iris Data의 Pair Plot")
    phase3_model.scatter_matrix()
    plt.show()
       
    # line graph using R ggplot2 function in Python environment by plotnine library
    from plotnine import *
    plotline.ggplot(phase3_model, mappig=plotline.aes(x='fcst_tm_hour', y='QGEN')) \
    + geom_line() \
    + ggtitle('Time Series Graph of QGEN')

    plotline.ggplot(phase3_model[phase3_model.QGEN >= 1], mapping=plotline.aes(x= phase3_model.QGEN_echo, y=phase3_model.QGEN))\
    + plotline.geom_point(colour ="red", size=0.5) + geom_smooth()\
    + plotline.geom_point(mapping=plotline.aes(x= phase3_model.QGEN_sejong1, y=phase3_model.QGEN), colour = "black", size=0.5)\
    + plotline.geom_point(mapping=plotline.aes(x= phase3_model.QGEN_sejong2, y=phase3_model.QGEN), colour = "blue", size=0.5)
        
    from pandas import scatter_matrix

    scatter_matrix(phase3_model, figsize=(10, 10))