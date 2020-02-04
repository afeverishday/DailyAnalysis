#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:07:04 2020

@author: keti
"""

#MongoDB 연동
import pymongo
import pandas as pd
import gc
import numpy as np
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pytictoc import TicToc
# 경고 출력하지 않음 -----------
import warnings
warnings.filterwarnings('ignore')

#working directory
import os
os.getcwd()

rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Tutorials/Regression/Car_Price/carprice_total.csv'))\
                      .rename(columns={'가격': 'price', '년식':'year','종류':'type','연비':'efficiency','마력':'power',
                                       '토크':'torque','연료':'fuel', '하이브리드':'hybrid','배기량':'displacement', '중량': 'weight' ,'변속기':'manual', })\
            .assign(hybrid=lambda x : x['hybrid'].astype('int'))\
            .replace('가솔린', 'gasoline').replace('디젤', 'diesel')\
            .replace('자동', 'auto').replace('수동', 'manual')\
            .replace('준중형', 'middle_small').replace('소형', 'small').replace('대형', 'big').replace('중형', 'middle')
#                      .assign(crtn_time=lambda x:pd.to_datetime(x['crtn_time'])).replace(np.nan, None)\
#                      .query("crtn_time> '2019-12-27 09:00' and crtn_time< '2019-12-28 09:00'")\
 #                     .drop('usage_mean', axis=1)
rawdata.nunique() 
rawdata.columns
rawdata.head().round(2)
rawdata.describe().round(2)
rawdata.info()
(rawdata.isnull().sum()/max(rawdata.count())).round(2)

##########데이터 전처리========================================================
    #rawdata = rawdata.dropna() #하나라도 비어 있으면 삭제
    #rawdata = rawdata.dropna(how='any') #하나라도 비어 있으면 삭제 (디폴트)
    #rawdata = rawdata.dropna(how='all') #모두 비어 있으면 삭제
    #rawdata = rawdata.dropna(thresh=1) #최소 1개는 값을 가져야 함

 # 결측값 대치 값
from sklearn.impute import SimpleImputer
transformer = SimpleImputer()
transformer.fit(rawdata) #SimpleImputer 모델에 df_x_train 데이터 적용 (평균값 계산)
rawdata = transformer.transform(rawdata) #트랜스포머의 transform() 함수는 결과를 넘파이 배열로 리턴

transformer = make_column_transformer(
            (SimpleImputer(), ['hour']),
            (SimpleImputer(strategy='median'), ['attendance']),
            remainder='passthrough'
            )
       
#    strategy='mean' 평균값으로 대치 (디폴트)
#    strategy='median' 중앙값으로 대치
#    strategy='most_frequent' 최빈값 (mode)으로 대치
#    strategy='constant', fill_value=1 특정값으로 대치, 예) transformer = SimpleImputer(strategy='constant', fill_value=1)
    

 # 이상치 대치
quartile_1 = rawdata.quantile(0.25)
quartile_3 = rawdata.quantile(0.75)
IQR = quartile_3 - quartile_1
condition = (rawdata < (quartile_1 - 1.5 * IQR)) | (rawdata > (quartile_3 + 1.5 * IQR))
condition = condition.any(axis=1)
rawdata_search = rawdata[condition]
print(rawdata_search)
    
rawdata = rawdata.drop(rawdata_search.index, axis=0)
   
def replace_attendance_outlier(value):
        quartile_1 = rawdata[value].quantile(0.25)
        quartile_3 = rawdata[value].quantile(0.75)
        IQR = quartile_3 - quartile_1

        if ((value < (quartile_1 - 1.5 * IQR)) | (value > (quartile_3 + 1.5 * IQR))):
            value = rawdata[value].median()
        return value
    
rawdata['attendance'] = rawdata['attendance'].apply(replace_attendance_outlier)
  


 ##########데이터 분석===========================================================  

# columns name sort
categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"]))
numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ])-set(['year']))
time_feature = list(set(rawdata.columns) - set(categorical_feature)-set(numerical_feature))
    
minmax_feature = list( set(numerical_feature)-set(['price']))
target_feature = list(['price'])
dummy_feature=list(set(categorical_feature))

import collections
collections.Counter(rawdata['type']) 
collections.Counter(rawdata['fuel'])
collections.Counter(rawdata['manual'])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['price'])],
                                                            rawdata['price'], test_size=0.3, random_state=777, stratify=rawdata['fuel'])
train_y=pd.DataFrame(train_y)
#train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['price'])], rawdata['price'], test_size=0.1, random_state=777, stratify=rawdata['year'])
    
train=pd.concat([train_x, train_y], axis=1) # columns bind
test=pd.concat([test_x, test_y], axis=1) # columns bind

tictoc = TicToc() #create instance of class
tictoc.tic() #Time elapsed since t.tic()
 # scaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
minmaxscaler = MinMaxScaler().fit(train_x[minmax_feature])
minmaxscaler_target = MinMaxScaler().fit(train_y)
train_minmax = pd.DataFrame(minmaxscaler.transform(train_x[minmax_feature]),
                                columns= minmax_feature, index=list(train_x.index.values)).round(2)
train_target = pd.DataFrame(minmaxscaler_target.transform(train_y),
                                columns= target_feature, index=list(train_y.index.values)).round(2)
train_dummy=pd.get_dummies(train_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
train_preprocess=pd.concat([train_minmax, train_dummy], axis=1) # columns bind
train_all=pd.concat([train_preprocess, train_target], axis=1)

#Correlattion=========================================
corr= train_all.drop((list(set(train_dummy))), axis=1).corr(method='pearson')\
        .loc[target_feature].round(3)
       
from minepy import MINE
def print_stats(mine):
    print("MIC:", round(mine.mic(),2)," MAS:", round(mine.mas(),2), " MEV:", round(mine.mev(),2), "MCN (eps=0):", round(mine.mcn(0),2), 
          "MCN (eps=1-MIC):", round(mine.mcn_general(),2), "GMIC:", round(mine.gmic(),2), " TIC:", round(mine.tic(),2))

index=0
plt.figure(figsize=(15, 10))
mine = MINE(alpha=0.6, c=15, est="mic_approx")
for i in minmax_feature:
    print(i)
    plt.subplot(611+index)
    index=index+1
    plt.scatter(train_all[i], train_all['price'])
    mine.compute_score(train_all[i], train_all['price'])
    print_stats(mine)



# corr and mic graph
plt.figure(figsize=(15, 10))
plt.subplot(611)
plt.scatter(train_all['torque'], train_all['price'])
mine.compute_score(train_all['torque'], train_all['price'])
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,0] ))
plt.subplot(612)
plt.scatter(train_all['weight'], train_all['price'])
mine.compute_score(train_all['weight'], train_all['price'])
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,1] ))
plt.subplot(233)
plt.scatter(train_all['power'], train_all['price'])
mine.compute_score(train_all['power'], train_all['price'])
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,2] ))
plt.subplot(234)
plt.scatter(train_all['displacement'], train_all['price'])
mine.compute_score(train_all['displacement'], train_all['price'])
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,3] ))
plt.subplot(235)
plt.scatter(train_all['efficiency'], train_all['price'])
mine.compute_score(train_all['efficiency'], train_all['price'])
plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,4] ))

train_x= train_preprocess
train_y= train_target


test_minmax = pd.DataFrame(minmaxscaler.transform(test_x[minmax_feature]),
                                columns= minmax_feature, index=list(test_x.index.values)).round(2)
test_target = pd.DataFrame(minmaxscaler_target.transform(pd.DataFrame(test_y)),
                                columns= target_feature, index=list(test_y.index.values)).round(2)
test_dummy=pd.get_dummies(test_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
test_x=pd.concat([test_minmax, test_dummy], axis=1) # columns bind
test_y=test_target


#GridSearchCV===============================================
from sklearn.model_selection import GridSearchCV,KFold
#  n_jobs= 4, 병렬 처리갯수? -1은 전부),   refit=True 좋은 estimator로 수정되어짐. 
from sklearn.linear_model import Ridge, Lasso, LinearRegression,ElasticNet
#    alpha : float, optional
#         디폴트 1.0,   클수록 가중치 규제 (특성 수 줄이기, 과대적합 방지),   작을수록 정확하게 (과대적합)
#    alpha 0이면 ordinary least square (일반적 리니어리그래션)
#           가중치가 커질수록 특성이 늘어나 훈련 데이터에 과적합됨   alpha 옵션을 크게 할수록 가중치 규제 (가중치의 크기가 못 커지게 하기, 과적합 방지)
#           크기 개념을 l1(맨하튼 거리) 으로   하냐 l2(유클리디안 거리의 제곱)로 생각하냐에 따라  라쏘(l1) 혹은 릿지(l2) 를 사용
param_grid = { 'alpha':np.arange(0,0.1,0.01) }
lasso_model = GridSearchCV( Lasso(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='r2',refit=True).fit(train_x, train_y)
print(lasso_model.best_params_)
print(lasso_model.best_score_)

param_grid = { 'alpha':np.arange(0,1,0.1) }
ridge_model = GridSearchCV( Ridge(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777),scoring='r2',refit=True).fit(train_x, train_y)
print(ridge_model.best_params_)
print(ridge_model.best_score_)

param_grid = { 'alpha':np.arange(0,1,0.1), 'l1_ratio':np.arange(0,1,0.1) }
elasticnet_model = GridSearchCV( ElasticNet(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='r2',refit=True).fit(train_x, train_y)
print(elasticnet_model.best_params_)
print(elasticnet_model.best_score_)

from sklearn.neural_network import MLPRegressor
param_grid = {'max_iter':np.arange(5,100,5), 'alpha':np.arange(0.1,1,0.1),'hidden_layer_sizes':np.arange(10,200,10)}
mlp_model = GridSearchCV(MLPRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(mlp_model.best_params_)
print(mlp_model.best_score_)

from sklearn.tree import DecisionTreeRegressor
#    criterion='mse', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'                     
param_grid = {'max_features':np.arange(1,5,1), 'max_depth':np.arange(1,15,1)}
tree_model = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(tree_model.best_params_)
print(tree_model.best_score_)

from sklearn.svm import LinearSVR, SVR
#    kernel : linear, poly,rbf
#    C : float, optional (default=1.0) 클수록 정확하게 (마진이 작아짐, 과대적합),  alpha (가중치 규제) 의 역수
#    gamma : float, optional (default=’auto’) 클수록 정확하게 (경사가 급해짐, 과대적합)   비선형 (kernel=‘rbf’, kernel=‘poly’)에서만 이 옵션 사용 
param_grid = {'kernel':['linear','rbf','poly'], 'C':np.arange(1,4,1)}
svr_model = GridSearchCV(SVR(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(svr_model.best_params_)
print(svr_model.best_score_)


from sklearn.ensemble import BaggingRegressor
#    base_estimator: base estimator to fit
#    n_estimators: the number of base estimators 
#    bootstrap : (default=True) 중복 할당 여부  True 베깅, S  False 페이스팅
param_grid = {'base_estimator':[Ridge(alpha=0),LinearRegression(fit_intercept=True),Lasso(alpha=0)], 'n_estimators':np.arange(2,20,2),'bootstrap':[True, 'S',False]}
bagging_model = GridSearchCV(BaggingRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(bagging_model.best_params_)
print(bagging_model.best_score_)

from sklearn.ensemble import RandomForestRegressor
#    n_estimators : integer, optional (default=10) The number of trees in the forest.
#    bootstrap : boolean, optional (default=True) True 베깅,   False 페이스팅
#    criterion : string, optional (default=”mse”) 'mse' (평균제곱오차),   'friedman_mse', 'mae'
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'
param_grid = {'n_estimators':np.arange(2,20,2),'bootstrap':[True,False],'max_features':np.arange(1,5,1), 'max_depth':np.arange(1,15,1)}
rf_model = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(rf_model.best_params_)
print(rf_model.best_score_)

from sklearn.ensemble import AdaBoostRegressor
#    base_estimator : object, optional (default=None)
#    n_estimators : integer, optional (default=50) The maximum number of estimators at which boosting is terminated.
param_grid = {'n_estimators':np.arange(10,100,10),'base_estimator':[Ridge(alpha=0),LinearRegression(fit_intercept=True),Lasso(alpha=0)]}
adaboost_model = GridSearchCV(AdaBoostRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(adaboost_model.best_params_)
print(adaboost_model.best_score_)

from sklearn.ensemble import GradientBoostingRegressor
#    n_estimators : int (default=100) The number of boosting stages to perform. 
#    criterion : string, optional (default=”friedman_mse”) friedman_mse, mse, mae
param_grid = {'n_estimators':np.arange(50,150,10)}
gradient_model = GridSearchCV(GradientBoostingRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
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
xgb_model = GridSearchCV(XGBRegressor(objective ='reg:squarederror'), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(xgb_model.best_params_)
print(xgb_model.best_score_)


from lightgbm import LGBMRegressor
param_grid = {'n_estimators':np.arange(10,50,10),'learning_rate':np.arange(0.001,0.01,0.001)}
lgb_model = GridSearchCV(LGBMRegressor(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(lgb_model.best_params_)
print(lgb_model.best_score_)

#validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

model_list=[linear_model,lasso_model,ridge_model,elasticnet_model,mlp_model,svr_model,tree_model,
            bagging_model,adaboost_model,gradient_model, rf_model,xgb_model,lgb_model]

for i in model_list:
    print(": ", cross_val_score(i, test_x, test_y, cv=StratifiedKFold(n_splits=3, random_state=777),scoring='mae')





result=test_x.assign(price=test_target,                     
                     price_linear= lambda x: np.where(linear_model.predict(test_x)<0, 0,
                                                   np.where(linear_model.predict(test_x)>1,1, linear_model.predict(test_x))),
                     price_lasso= lambda x: np.where(lasso_model.predict(test_x)<0, 0,
                                                   np.where(lasso_model.predict(test_x)>1,1, lasso_model.predict(test_x))),                                                     
                     price_ridge= lambda x: np.where(ridge_model.predict(test_x)<0, 0,
                                                   np.where(ridge_model.predict(test_x)>1,1, ridge_model.predict(test_x))),
                     price_elasticnet= lambda x: np.where(elasticnet_model.predict(test_x)<0, 0,
                                                   np.where(elasticnet_model.predict(test_x)>1,1, elasticnet_model.predict(test_x))),                                                     
                     price_linsvc= lambda x: np.where(linsvc_model.predict(test_x)<0, 0,
                                                   np.where(linsvc_model.predict(test_x)>1,1, linsvc_model.predict(test_x))), 
                     price_mlp= lambda x: np.where(mlp_model.predict(test_x)<0, 0,
                                                   np.where(mlp_model.predict(test_x)>1,1, mlp_model.predict(test_x))),                                   
                     price_tree= lambda x: np.where(tree_model.predict(test_x)<0,0,
                                                np.where(tree_model.predict(test_x)>1,1,tree_model.predict(test_x))),
                     price_bagging= lambda x: np.where(bagging_model.predict(test_x)<0,0,
                                                np.where(bagging_model.predict(test_x)>1,1,bagging_model.predict(test_x))),
                     price_adaboost= lambda x: np.where(adaboost_model.predict(test_x)<0,0,
                                                np.where(adaboost_model.predict(test_x)>1,1,adaboost_model.predict(test_x))),                  
                     price_rf= lambda x: np.where(rf_model.predict(test_x)<0,0,
                                                np.where(rf_model.predict(test_x)>1,1,rf_model.predict(test_x))),
                     price_gradient= lambda x: np.where(gradient_model.predict(test_x)<0,0,
                                                np.where(gradient_model.predict(test_x)>1,1,gradient_model.predict(test_x))),                                                 
                     price_xgb= lambda x: np.where(xgb_model.predict(test_x)<0,0,
                                                np.where(xgb_model.predict(test_x)>1,1,xgb_model.predict(test_x))),
                     price_lgb= lambda x: np.where(lgb_model.predict(test_x)<0,0,
                                                np.where(lgb_model.predict(test_x)>1,1,lgb_model.predict(test_x))),
                     price_inverse=lambda x: minmaxscaler_target.data_max_*(x['price'])+minmaxscaler_target.data_min_,
                     price_linear_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_linear']) + minmaxscaler_target.data_min_,
                     price_lasso_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_lasso'])+minmaxscaler_target.data_min_,
                     price_elasticnet_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_elasticnet'])+minmaxscaler_target.data_min_,
                     price_ridge_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_ridge'])+minmaxscaler_target.data_min_,
                     price_linsvc_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_linsvc']) + minmaxscaler_target.data_min_,
                     price_mlp_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_mlp']) + minmaxscaler_target.data_min_,
                     price_tree_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_tree'])+minmaxscaler_target.data_min_,
                     price_bagging_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_bagging'])+minmaxscaler_target.data_min_,
                     price_adaboost_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_adaboost']) + minmaxscaler_target.data_min_,
                     price_rf_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_rf'])+minmaxscaler_target.data_min_,
                     price_gradient_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_gradient'])+minmaxscaler_target.data_min_,
                     price_lgb_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_lgb'])+minmaxscaler_target.data_min_,
                     price_xgb_inverse=lambda x: minmaxscaler_target.data_max_*(x['price_xgb'])+minmaxscaler_target.data_min_).drop(set(test_dummy),axis=1)


from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
def precision(data,predict,origin):
    Rsquared = r2_score(data[origin],data[predict]).round(3)
    MAE = mean_absolute_error(data[origin],data[predict]).round(3)
    MSE = mean_squared_error(data[origin],data[predict]).round(3)
    RMSE = np.sqrt(mean_squared_error(data[origin],data[predict])).round(3)
    MSLE = mean_squared_log_error(data[origin],data[predict]).round(3)
    RMLSE = np.sqrt(mean_squared_log_error(data[origin],data[predict])).round(3)
    MAPE = round(np.mean((abs(data[origin]-data[predict]))/(data[origin]))*100,2)
    sMAPE = round(np.mean(200*(abs(data[origin]-data[predict]))/(data[origin]+data[predict])),2)
    print(predict,'[Rsquared:', Rsquared, 'MAE:',MAE, 'MSE:',MSE, 'RMSE:',RMSE, 'MSLE:', MSLE,'RMLSE:',RMLSE,'MAPE:',MAPE,'sMAPE:',sMAPE,']')
    

score_model1=['price_linear','price_lasso','price_elasticnet','price_ridge','price_linsvc','price_mlp','price_tree','price_bagging','price_adaboost','price_rf','price_gradient','price_lgb','price_xgb' ]
score_model2=['price_linear_inverse','price_lasso_inverse','price_elasticnet_inverse','price_mlp_inverse','price_ridge_inverse','price_linsvc_inverse','price_tree_inverse',
              'price_bagging_inverse','price_adaboost_inverse','price_rf_inverse','price_gradient_inverse','price_lgb_inverse', 'price_xgb_inverse']

for i in score_model2:
    precision(result.query('price_inverse>1'),i,'price_inverse')
    precision(result,i,'price_inverse')
        
for i in score_model1:
    precision(result,i,'price')    