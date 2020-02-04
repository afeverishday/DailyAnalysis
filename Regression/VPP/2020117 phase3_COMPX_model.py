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
# time = sys.argv[1]

def phase3_model_COMPX(crtntime):
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
             "QGEN":True,"VOLUME":True, "SRAD_group":True })))\
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
             "FCST_SRAD_SLOPE":True, "FCST_SRAD_HORIZ":True,"FCST_TEMP" :True, "FCST_MTEMP":True})))\
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
                 .assign(COMPX_ID=lambda x: x['COMPX_ID'].astype('str'),
                         fcst_tm_hour=lambda x: x['fcst_tm_hour'].astype('str'),
                         SRAD_group=lambda x: x['SRAD_group'].astype('int'))
                 
#                 .drop(['SRAD_group','CRTN_TM','LEAD_HR','fcst_tm_hour'], axis=1)


    len(rawdata.COMPX_ID.unique())

    rawdata.columns
    rawdata.head().round(2)
    rawdata.describe().round(2)
    rawdata.info()
    (rawdata.isnull().sum()/max(rawdata.count())).round(2)
    
    from sklearn.model_selection import train_test_split
#    train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['QGEN'])],\
#                                                         rawdata['QGEN'], test_size=0.1, random_state=777, stratify=rawdata['COMPX_ID'])
#    
#    train=pd.concat([train_x, train_y], axis=1) # columns bind
#    test=pd.concat([test_x, test_y], axis=1) # columns bind
 
    import collections
    collections.Counter(rawdata['COMPX_ID']) 
    collections.Counter(rawdata['fcst_tm_hour']) 
  
           ##########데이터 분석
    print(rawdata['COMPX_ID'].sort_values())
    
    # columns name sort
    categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"])-set(['COMPX_ID']))
    numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ])-set(['COMPX_ID','LEAD_HR','CRTN_TM']))
    time_feature= list(set(rawdata.columns) - set(categorical_feature)-set(numerical_feature)-set(['COMPX_ID']))
    index_feature= list(set(['COMPX_ID']))
        
#    minmax_feature = list( set(numerical_feature)-set(['FCST_MTEMP','FCST_TEMP','HFSFC',  'VOLUME', 'QGEN', 'QGEN_sejong1','QGEN_sejong2', 'QGEN_echo', 'QGEN_keti']))
#    volume_feature = list(['QGEN', 'QGEN_sejong1','QGEN_sejong2', 'QGEN_echo', 'QGEN_keti','VOLUME'])
#    standard_feature = list(set(numerical_feature)-set(minmax_feature)-set(volume_feature))
#    dummy_feature=list(set(categorical_feature)- set(['COMPX_ID']))
    
    minmax_feature = list( set(numerical_feature)-set(['FCST_MTEMP','FCST_TEMP','HFSFC', 'QGEN']))
    target_feature = list(['QGEN'])
    standard_feature = list(set(numerical_feature)-set(minmax_feature)-set(target_feature))
    dummy_feature=list(set(categorical_feature))
       
    tictoc.tic() #Time elapsed since t.tic()
        # 컬럼만 있는 데이터프레임 생성
    corr_COMPX = pd.DataFrame(columns=list(set(rawdata)-set(['COMPX_ID', 'CRTN_TM','LEAD_HR', 'fcst_tm_hour'])))
    phase3_COMPX = pd.DataFrame(columns=['QGEN_echo', 'QGEN_sejong2', 'QGEN_sejong1', 'QGEN_keti',
                                         'COMPX_ID', 'QGEN','QGEN_linear', 'QGEN_ridge',
                                         'QGEN_tree','QGENlinsvc' ,'QGEN_bagging','QGEN_rf', 'QGEN_gradient', 'QGEN_xgb'])
    feature_list=[minmax_feature, target_feature, standard_feature, dummy_feature]
    model=[]
    model_list=[]
    #model_score={}
    # scaler
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
    import xgboost
    import lightgbm as lgb
    from lightgbm import LGBMModel,LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import LinearSVR, SVR
    
    for COMPX_ID,data in rawdata.groupby('COMPX_ID'):
        
        train_x, test_x, train_y, test_y = train_test_split(data[set(data.columns)-set(['QGEN'])],\
                                                            data['QGEN'], test_size=0.3, random_state=1295)
#        train=pd.concat([train_x, train_y], axis=1) # columns bind
#        test=pd.concat([test_x, test_y], axis=1) # columns bind
        train_y=pd.DataFrame(train_y)
        test_y=pd.DataFrame(test_y)
        print(COMPX_ID)
        data.head(5)
    
        globals()['minmaxscaler_{}'.format(COMPX_ID)]= MinMaxScaler().fit(train_x[minmax_feature])
        globals()['minmaxscaler_target_{}'.format(COMPX_ID)]= MinMaxScaler().fit(train_y)
        globals()['standardscaler_{}'.format(COMPX_ID)]= StandardScaler().fit(train_x[standard_feature])
        
        minmaxscaler = globals()['minmaxscaler_{}'.format(COMPX_ID)].fit(train_x[minmax_feature])
        minmaxscaler_target = globals()['minmaxscaler_target_{}'.format(COMPX_ID)].fit(train_y)
        standardscaler = globals()['standardscaler_{}'.format(COMPX_ID)].fit(train_x[standard_feature])
        
        train_minmax = pd.DataFrame(minmaxscaler.transform(train_x[minmax_feature]),
                                    columns= minmax_feature, index=list(train_x.index.values)).round(2)
        train_target = pd.DataFrame(minmaxscaler_target.transform(train_y),
                                    columns= target_feature, index=list(train_y.index.values)).round(2)
        train_standard= pd.DataFrame(standardscaler.transform(train_x[standard_feature]),
                                     columns=standard_feature,index=list(train_x.index.values)).round(2)   
        train_dummy=pd.get_dummies(train_x['fcst_tm_hour'], prefix='fcst_tm_hour', drop_first=True)
        
        train_preprocess=pd.concat([train_minmax, train_standard, train_dummy], axis=1) # columns bind
        train_all=pd.concat([train_preprocess, train_target], axis=1)
        
        corr= train_all.drop((list(set(train_dummy))), axis=1)\
        .corr(method='pearson')\
        .assign(COMPX_ID=COMPX_ID).iloc[[15]].round(3)
        corr_COMPX= pd.concat([corr_COMPX, corr], axis=0).drop('QGEN', axis=1)
        
               
        train_x= train_preprocess
        train_y= train_target
                
        globals()['linear_model_{}'.format(COMPX_ID)] =LinearRegression(fit_intercept=True).fit(train_x, train_y)
#        globals()['lasso_model_{}'.format(COMPX_ID)]=Lasso(alpha=0.5).fit(train_x, train_y)
        globals()['ridge_model_{}'.format(COMPX_ID)] =Ridge(alpha=0.5).fit(train_x, train_y)
        globals()['tree_model_{}'.format(COMPX_ID)]= DecisionTreeRegressor().fit(train_x, train_y) 
        globals()['linsvc_model_{}'.format(COMPX_ID)]= LinearSVR().fit(train_x, train_y) 
#        globals()['svc_model_{}'.format(COMPX_ID)]=SVC(kernel='linear') .fit(train_x, train_y) 
        globals()['bagging_model_{}'.format(COMPX_ID)] = BaggingRegressor(base_estimator=Ridge(alpha=0.5)).fit(train_x, train_y)
        globals()['rf_model_{}'.format(COMPX_ID)] = RandomForestRegressor().fit(train_x, train_y)
#        globals()['adaboost_model_{}'.format(COMPX_ID)] = AdaBoostRegressor(base_estimator=Lasso()).fit(train_x, train_y)
        globals()['gradient_model_{}'.format(COMPX_ID)] = GradientBoostingRegressor(criterion='mae').fit(train_x, train_y)
        globals()['xgb_model_{}'.format(COMPX_ID)]=xgboost.XGBRegressor(objective ='reg:squarederror').fit(train_x, train_y)
        globals()['lgb_model_{}'.format(COMPX_ID)]=lgb.LGBMRegressor().fit(train_x, train_y)

        model.extend([globals()['minmaxscaler_{}'.format(COMPX_ID)],globals()['minmaxscaler_target_{}'.format(COMPX_ID)],globals()['standardscaler_{}'.format(COMPX_ID)], 
                      globals()['linear_model_{}'.format(COMPX_ID)], globals()['ridge_model_{}'.format(COMPX_ID)], globals()['tree_model_{}'.format(COMPX_ID)],
                      globals()['linsvc_model_{}'.format(COMPX_ID)], globals()['bagging_model_{}'.format(COMPX_ID)], globals()['rf_model_{}'.format(COMPX_ID)],
                      globals()['gradient_model_{}'.format(COMPX_ID)], globals()['xgb_model_{}'.format(COMPX_ID)],globals()['lgb_model_{}'.format(COMPX_ID)] ] )
        model_list.extend(['minmaxscaler_{}'.format(COMPX_ID),'minmaxscaler_target_{}'.format(COMPX_ID),'standardscaler_{}'.format(COMPX_ID),
                           'linear_model_{}'.format(COMPX_ID), 'ridge_model_{}'.format(COMPX_ID),'tree_model_{}'.format(COMPX_ID),
                           'linsvc_model_{}'.format(COMPX_ID),'bagging_model_{}'.format(COMPX_ID),'rf_model_{}'.format(COMPX_ID),
                           'gradient_model_{}'.format(COMPX_ID), 'xgb_model_{}'.format(COMPX_ID),'lgb_model_{}'.format(COMPX_ID)])
        
        test_minmax = pd.DataFrame(minmaxscaler.transform(test_x[minmax_feature]), columns= minmax_feature, index=list(test_x.index.values)).round(2)
        test_target =  pd.DataFrame(minmaxscaler_target.transform(test_y), columns= target_feature, index=list(test_y.index.values)).round(2)
        test_standard= pd.DataFrame(standardscaler.transform(test_x[standard_feature]), columns=standard_feature,index=list(test_x.index.values)).round(2)   
        test_dummy=pd.get_dummies(test_x['fcst_tm_hour'], prefix='fcst_tm_hour', drop_first=True)
        test_preprocess=pd.concat([test_minmax, test_standard, test_dummy], axis=1) # columns bind
        test_x= test_preprocess
        test_y= test_target

##        score = make_scorer(r2_score) 
#        score = make_scorer(mean_absolute_error)
#        score_COMPX={'linear_model_{}'.format(COMPX_ID): score(globals()['linear_model_{}'.format(COMPX_ID)], test_x, test_y).round(3),
#                     'lasso_model_{}'.format(COMPX_ID):score(globals()['lasso_model_{}'.format(COMPX_ID)], test_x, test_y).round(3),
#                     'ridge_model_{}'.format(COMPX_ID):score(globals()['ridge_model_{}'.format(COMPX_ID)], test_x, test_y).round(3),
#                     'bagging_model_{}'.format(COMPX_ID):score(globals()['bagging_model_{}'.format(COMPX_ID)], test_x, test_y).round(3),
#                     'tree_model_{}'.format(COMPX_ID):score(globals()['tree_model_{}'.format(COMPX_ID)], test_x, test_y).round(3),
#                     'gradient_model_{}'.format(COMPX_ID):score(globals()['gradient_model_{}'.format(COMPX_ID)], test_x, test_y).round(3),
#                     'xgb_model_{}'.format(COMPX_ID):score(globals()['xgb_model_{}'.format(COMPX_ID)], test_x, test_y).round(3)}
        
#        model_score.update(score_COMPX)
#        del(score_COMPX)

        result= test_x\
            .assign(COMPX_ID=COMPX_ID,        
                    QGEN=test_y,
                    QGEN_echo=lambda x: x[['QGEN_echo']],
                    QGEN_sejong1=lambda x: x[['QGEN_sejong1']],
                    QGEN_sejong2=lambda x: x[['QGEN_sejong2']],
                    QGEN_keti=lambda x: x[['QGEN_keti']],
                    QGEN_linear= lambda x: np.where( (globals()['linear_model_{}'.format(COMPX_ID)].predict(test_x)) <0,0,
                                                    np.where( (globals()['linear_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['linear_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_ridge= lambda x: np.where( (globals()['ridge_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['ridge_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['ridge_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_tree= lambda x: np.where( (globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_linsvc= lambda x: np.where( (globals()['linsvc_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['linsvc_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['linsvc_model_{}'.format(COMPX_ID)].predict(test_x))),                                                  
                    QGEN_bagging= lambda x: np.where( (globals()['bagging_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['bagging_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['bagging_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_rf= lambda x: np.where( (globals()['rf_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['rf_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['rf_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_gradient= lambda x: np.where( (globals()['gradient_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['gradient_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['gradient_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_xgb= lambda x: np.where( (globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x))),
                    QGEN_lgb= lambda x: np.where( (globals()['lgb_model_{}'.format(COMPX_ID)].predict(test_x))<0,0,
                                                    np.where( (globals()['lgb_model_{}'.format(COMPX_ID)].predict(test_x))>1,1,
                                                              globals()['lgb_model_{}'.format(COMPX_ID)].predict(test_x))),                                                 
                    QGEN_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_linear_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_linear'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_ridge_inverse=lambda x:  (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_ridge'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_tree_inverse=lambda x:  (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_tree'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_linsvc_inverse=lambda x:  (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_linsvc'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_bagging_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_bagging'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_rf_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_rf'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_gradient_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_gradient'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_xgb_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_xgb'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_),
                    QGEN_lgb_inverse=lambda x: (globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_max_*(x['QGEN_lgb'])+globals()['minmaxscaler_target_{}'.format(COMPX_ID)].data_min_))\
                                        .drop(set(test_dummy)|set(['NDNSW', 'FCST_SRAD_SLOPE',  'FCST_SRAD_HORIZ', 
                           'TMP-SFC',  'TMP', 'RH', 'FCST_MTEMP', 'HFSFC', 'FCST_TEMP']), axis=1)


#                    QGEN_lasso= lambda x: globals()['lasso_model_{}'.format(COMPX_ID)].predict(x).reshape(len(x),1),
#                    QGEN_ridge= lambda x: globals()['ridge_model_{}'.format(COMPX_ID)].predict(x).reshape(len(x),1),
#                    QGEN_tree= lambda x: globals()['tree_model_{}'.format(COMPX_ID)].predict(x).reshape(len(x),1),
#                    QGEN_bagging= lambda x: globals()['bagging_model_{}'.format(COMPX_ID)].predict(x).reshape(len(x),1),
#                    QGEN_gradient= lambda x: globals()['gradient_model_{}'.format(COMPX_ID)].predict(x).reshape(len(x),1),
#                    QGEN_xgb= lambda x: globals()['xgb_model_{}'.format(COMPX_ID)].predict(x).reshape(len(x),1)
#                    QGEN=minmaxscaler_target.inverse_transform(test_y),
#                    QGEN_echo=lambda x: minmaxscaler_target.inverse_transform(x[['QGEN_echo']]),
#                    QGEN_sejong1=lambda x: minmaxscaler_target.inverse_transform(x[['QGEN_sejong1']]),
#                    QGEN_sejong2=lambda x: minmaxscaler_target.inverse_transform(x[['QGEN_sejong2']]),
#                    QGEN_keti=lambda x: minmaxscaler_target.inverse_transform(x[['QGEN_keti']]),
#                    QGEN_linear= minmaxscaler_target.inverse_transform(globals()['linear_model_{}'.format(COMPX_ID)].predict(test_x)),
#                    QGEN_lasso= minmaxscaler_target.inverse_transform(globals()['lasso_model_{}'.format(COMPX_ID)].predict(test_x).reshape(len(test_x),1)),
#                    QGEN_ridge= minmaxscaler_target.inverse_transform(globals()['ridge_model_{}'.format(COMPX_ID)].predict(test_x).reshape(len(test_x),1)),
#                    QGEN_tree= minmaxscaler_target.inverse_transform(globals()['tree_model_{}'.format(COMPX_ID)].predict(test_x).reshape(len(test_x),1)),
#                    QGEN_bagging= minmaxscaler_target.inverse_transform(globals()['bagging_model_{}'.format(COMPX_ID)].predict(test_x).reshape(len(test_x),1)),
#                    QGEN_gradient= minmaxscaler_target.inverse_transform(globals()['gradient_model_{}'.format(COMPX_ID)].predict(test_x).reshape(len(test_x),1)),
#                    QGEN_xgb= minmaxscaler_target.inverse_transform(globals()['xgb_model_{}'.format(COMPX_ID)].predict(test_x).reshape(len(test_x),1)),
#                    QGEN_linear= lambda x : np.where(x['QGEN_linear']<0,0,x['QGEN_linear']),
#                    QGEN_lasso= lambda x : np.where(x['QGEN_lasso']<0,0,x['QGEN_lasso']),
#                    QGEN_ridge= lambda x : np.where(x['QGEN_ridge']<0,0,x['QGEN_ridge']),
#                    QGEN_tree= lambda x : np.where(x['QGEN_tree']<0,0,x['QGEN_tree']),
#                    QGEN_bagging= lambda x : np.where(x['QGEN_bagging']<0,0,x['QGEN_bagging']),
#                    QGEN_gradient= lambda x : np.where(x['QGEN_gradient']<0,0,x['QGEN_gradient']),
#                    QGEN_xgb= lambda x : np.where(x['QGEN_xgb']<0,0,x['QGEN_xgb'])
            
        phase3_COMPX=pd.concat([phase3_COMPX,result], axis=0) # raw bind
    
     
    tictoc.toc("train_model") #Time elapsed since t.tic()




    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

    def precision(data,predict,origin):
        Rsquared = r2_score(data[origin],data[predict]).round(2)
        MAE = mean_absolute_error(data[origin],data[predict]).round(2)
        MSE = mean_squared_error(data[origin],data[predict]).round(2)
        RMSE = np.sqrt(mean_squared_error(data[origin],data[predict])).round(2)
        MSLE = mean_squared_log_error(data[origin],data[predict]).round(2)
        RMLSE = np.sqrt(mean_squared_log_error(data[origin],data[predict])).round(2)
        MAPE = np.mean((abs(data[origin]-data[predict]))/(data[origin]))
        sMAPE = np.mean(200*(abs(data[origin]-data[predict]))/(data[origin]+data[predict]))
        print(predict,'[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,'MAPE:',MAPE,'sMAPE:',sMAPE,']')
    
score_model1=['QGEN_linear','QGEN_ridge','QGEN_tree','QGEN_linsvc','QGEN_bagging','QGEN_rf','QGEN_gradient','QGEN_xgb','QGEN_lgb']

score_model2=['QGEN_linear_inverse', 'QGEN_ridge_inverse','QGEN_tree_inverse','QGEN_linsvc_inverse','QGEN_bagging_inverse','QGEN_rf_inverse','QGEN_gradient_inverse','QGEN_xgb_inverse','QGEN_lgb_inverse']

score_model3=['QGEN_echo','QGEN_sejong1','QGEN_sejong2','QGEN_keti']
for i in score_model1:
    precision(phase3_COMPX,i,'QGEN')

for i in score_model2:
    precision(phase3_COMPX,i,'QGEN_inverse')

for i in score_model3:
    precision(rawdata,i,'QGEN')

    
# save model   
from sklearn import datasets
from sklearn.externals import joblib
joblib.dump(model,'/home/keti/vpp/vpp_model/model_COMPX_ID.pkl' )
joblib.dump(model_list,'/home/keti/vpp/vpp_model/model_list_COMPX_ID.pkl' )
joblib.dump(feature_list,'/home/keti/vpp/vpp_model/feature_list_COMPX_ID.pkl' )

    #multi model load
    tictoc.tic() #Time elapsed since t.tic()
    for i in np.arange(0,len(model_list),1):
        print(i)
        globals()[model_list[i]]=joblib.load('/home/keti/vpp/vpp_model/model_COMPX_ID.pkl')[i]

    tictoc.toc("read_model") #Time elapsed since t.tic()

