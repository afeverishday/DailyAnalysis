#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:55:13 2020

@author: keti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rawdata = pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Tutorials/Multiple_Classification/Loan/Loan payments data.csv'))\
            .assign(effective_date=lambda x:pd.to_datetime(x['effective_date']),
                    due_date=lambda x:pd.to_datetime(x['due_date']),
                    paid_off_time=lambda x:pd.to_datetime(x['paid_off_time']),
                    past_due_days= lambda x:x['past_due_days'].replace(np.nan,0),
                    loan_status=lambda x: x[['loan_status']].applymap({'PAIDOFF':0, 'COLLECTION_PAIDOFF':1,'COLLECTION':2}.get) )

# EDA=====================================================
rawdata.columns
rawdata.head()
rawdata.describe()
rawdata.info()
rawdata.isnull().sum()
(rawdata.isnull().sum()/max(rawdata.count())).round(2)

import collections
collections.Counter(rawdata['education']) 
collections.Counter(rawdata['Gender'])

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


# columns name sort
categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"])-set(['Loan_ID']))
numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ]))
time_feature = list(set(rawdata.columns) - set(categorical_feature)-set(numerical_feature)-set(['Loan_ID']))

minmax_feature = list( set(numerical_feature)-set(['loan_status']))
target_feature = list(['loan_status'])
dummy_feature=list(set(categorical_feature))

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['loan_status'])],
                                                            rawdata['loan_status'], test_size=0.3, random_state=777, stratify=rawdata['loan_status'])
train_y=pd.DataFrame(train_y)
test_y=pd.DataFrame(test_y)
#train_x, test_x, train_y, test_y = train_test_split(rawdata[set(rawdata.columns)-set(['price'])], rawdata['price'], test_size=0.1, random_state=777, stratify=rawdata['year'])
    
train=pd.concat([train_x, train_y], axis=1) # columns bind
test=pd.concat([test_x, test_y], axis=1) # columns bind

from pytictoc import TicToc
tictoc = TicToc() #create instance of class
tictoc.tic() #Time elapsed since t.tic()
 # scaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
minmaxscaler = MinMaxScaler().fit(train_x[minmax_feature])
train_minmax = pd.DataFrame(minmaxscaler.transform(train_x[minmax_feature]),
                                columns= minmax_feature, index=list(train_x.index.values)).round(2)
train_target = pd.get_dummies(train_y[target_feature], prefix='loan_status', drop_first=False)
train_dummy=pd.get_dummies(train_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
train_preprocess=pd.concat([train_minmax, train_dummy], axis=1) # columns bind
train_all=pd.concat([train_preprocess, train_target], axis=1)

#Correlattion=========================================
# columns name sort
for col in categorical_feature:
    unique_list = rawdata[col].unique()
    print(unique_list)

for col in categorical_feature:
    rawdata[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

for col in numerical_feature:
    sns.distplot(rawdata.loc[rawdata[col].notnull(), col])
    plt.title(col)
    plt.show()

sns.pairplot(rawdata[list(numerical_feature)], hue='loan_status', 
             x_vars=numerical_feature, y_vars=numerical_feature)
plt.show()

for col in numerical_feature:
    plt.figure(figsize=(12,6))
    sns.boxplot(x='Gender', y=col, hue='loan_status', data=rawdata.dropna())
    plt.title("Gender - {}".format(col))
    plt.show()

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
    plt.scatter(train_all[i], train_all['loan_status'])
    mine.compute_score(train_all[i], train_all['loan_status'])
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
test_target = pd.get_dummies(test_y[target_feature], prefix='loan_status', drop_first=False)
test_dummy=pd.get_dummies(test_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
test_x=pd.concat([test_minmax, test_dummy], axis=1) # columns bind
test_y=test_target


#GridSearchCV===============================================
from sklearn.model_selection import GridSearchCV,KFold
#  n_jobs= 4, 병렬 처리갯수? -1은 전부),   refit=True 좋은 estimator로 수정되어짐. 

from sklearn.linear_model import LogisticRegression
# penalty : str, ‘l1’, ‘l2’, ‘elasticnet’ or ‘none’, optional (default=’l2’), 가중치 규제 (특성 수 줄이기, 과대적합 방지)
#  l1: 맨하튼 거리, 오차 = 오차 + alpha * (|w1| + |w2|), l2: 유클리디안 거리의 제곱, 오차 = 오차 + alpha * (W1^2 + w2^2)
# C : float, optional (default=1.0) ,alpha 의 역수,  클수록 가중치 규제, 작을수록 정확하게 (과적합)
param_grid = { 'penalty' :['l1','l2'],'C':np.arange(1,10,1),'max_iter':np.arange(10,100,10) }
logistic_model = GridSearchCV( LogisticRegression(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='accuracy',refit=True).fit(train_x, train_y)
print(logistic_model.best_params_)
print(logistic_model.best_score_)

from sklearn.neural_network import MLPClassifier
param_grid = {'max_iter':np.arange(5,100,5), 'alpha':np.arange(0.1,1,0.1),'hidden_layer_sizes':np.arange(10,200,10),'activation':[‘identity’, ‘logistic’, ‘tanh’, ‘relu’]}
mlp_model = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(mlp_model.best_params_)
print(mlp_model.best_score_)

from sklearn.neighbors import KNeighborsClassifier
# n_neighbors, which has been metioned earlier
# weights : ‘uniform’(same weight), ‘distance’(closer points more heavily weighted)
# metric: 'manhattan', 'euclidean' how the distance of neighboring points 
param_grid = { 'metric' :['manhattan','euclidean'],'n_neighbors':np.arange(1,10,1),'weights':['distance','uniform'] }
knn_model = GridSearchCV( KNeighborsClassifier(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='accuracy',refit=True).fit(train_x, train_y)
print(knn_model.best_params_)
print(knn_model.best_score_)

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB().fit(train_x, train_y)

from sklearn.tree import DecisionTreeClassifier
   # criterion : 'gini', 'entropy'(default=”gini”) split 할 특성 선택 알고리즘, 어떤 기준으로 정보 획득량을 계산해 가지를 분리 할 것인지
   # max_depth : int(default=None) 얼마나 깊게 트리를 만들어 갈거냐   None 이면 최대한 깊게 (불순도 혹은 복잡도가 0일 때까지)   클수록 정확하게 (과대적합),   작을수록 가지치기 (과대적합 방지)
   # max_leaf_nodes : int(default=None) 최대 몇개 잎 노드가 만들어 질때 까지  split(하위 (잎) 노드로 분리) 할 것이냐   클수록 정확하게 (과대적합),   작을수록 가지치기 (과대적합 방지)
   # min_samples_split : float(default=2) 샘플이 최소한 몇개 이상이어야  split(하위 (잎) 노드로 분리) 할거냐
   #  int일 경우 주어진 값을 그대로 사용, float일 경우 0에서 1사이의 값을 줄 수 있으며 전체 데이터 수*min_sample_split의 값을 사용 # 클수록 가지치기 (과대적합 방지),   작을수록 정확하게 (과대적합)   
   # min_samples_leaf : int, float, optional (default=1) (잎) 노드가 되려면 가지고 있어야할 최소 샘플 수     #  클수록 가지치기 (과대적합 방지),  작을수록 정확하게 (과대적합)
param_grid = {'criterion' :['gini','entropy'],'max_features':np.arange(1,5,1), 'max_depth':np.arange(1,15,1)}
tree_model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(tree_model.best_params_)
print(tree_model.best_score_)

from sklearn.svm import SVC
#    kernel : linear, poly,rbf
#    C : float, optional (default=1.0) 클수록 정확하게 (마진이 작아짐, 과대적합),  alpha (가중치 규제) 의 역수
#    gamma : float, optional (default=’auto’) 클수록 정확하게 (경사가 급해짐, 과대적합)   비선형 (kernel=‘rbf’, kernel=‘poly’)에서만 이 옵션 사용 
param_grid = {'kernel':['linear','rbf','poly'], 'C':np.arange(1,10,1)}
svm_model = GridSearchCV(SVC(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(svm_model.best_params_)
print(svm_model.best_score_)

from sklearn.ensemble import VotingClassifier
# voting, hard: 직접 투표 , 결과에서 가장 많이 나온 결과를 채택, 디폴트  soft: 간접 투표 ,  각 클래스 확률을 합한 것들 중 가장 큰 것를 채택
# weights : array-like, shape = [n_classifiers], optional (default=`None`)  ex) weights=[2,1,1]    
voting_model1 = VotingClassifier(estimators=[('lr', LogisticRegression()), 
                                             ('kn', KNeighborsClassifier())],
                                              voting='hard', weights=None).fit(train_x, train_y)
voting_model2 = VotingClassifier(estimators=[('lr', LogisticRegression()), 
                                             ('kn', KNeighborsClassifier())],
                                              voting='soft', weights=None).fit(train_x, train_y)  

from sklearn.ensemble import BaggingClassifier
#    base_estimator: base estimator to fit
#    n_estimators: the number of base estimators 
#    bootstrap : (default=True) 중복 할당 여부  True 베깅, S  False 페이스팅
param_grid = {'base_estimator':[DecisionTreeClassifier(),LogisticRegression(),KNeighborsClassifier()], 'n_estimators':np.arange(2,20,2),'bootstrap':[True, 'S',False]}
bagging_model = GridSearchCV(BaggingClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(bagging_model.best_params_)
print(bagging_model.best_score_)


from sklearn.ensemble import RandomForestClassifier
#    n_estimators : integer, optional (default=10) The number of trees in the forest.
#    bootstrap : boolean, optional (default=True) True 베깅,   False 페이스팅
#    criterion : string, optional (default=”mse”) 'mse' (평균제곱오차),   'friedman_mse', 'mae'
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'
param_grid = {'n_estimators':np.arange(2,10,1),'bootstrap':[True,False],'max_features':np.arange(1,8,1), 'max_depth':np.arange(1,10,1)}
rf_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(rf_model.best_params_)
print(rf_model.best_score_)

from sklearn.ensemble import AdaBoostClassifier
#    base_estimator : object, optional (default=None)
#    n_estimators : integer, optional (default=50) The maximum number of estimators at which boosting is terminated.
param_grid = {'n_estimators':np.arange(10,100,10),'base_estimator':[DecisionTreeClassifier(),LogisticRegression()]}
adaboost_model = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(adaboost_model.best_params_)
print(adaboost_model.best_score_)

from sklearn.ensemble import GradientBoostingClassifier
#    n_estimators : int (default=100) The number of boosting stages to perform. 
#    criterion : string, optional (default=”friedman_mse”) friedman_mse, mse, mae
param_grid = {'n_estimators':np.arange(30,50,10)}
gradient_model = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(gradient_model.best_params_)
print(gradient_model.best_score_)

from lightgbm import LGBMModel,LGBMClassifier
param_grid = {'n_estimators':np.arange(10,50,10),'learning_rate':np.arange(0.001,0.01,0.001)}
lgb_model = GridSearchCV(LGBMClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(lgb_model.best_params_)
print(lgb_model.best_score_)

from xgboost import XGBClassifier
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
xgb_model = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=KFold(n_splits=2, random_state=777),refit=True).fit(train_x, train_y)
print(xgb_model.best_params_)
print(xgb_model.best_score_)

   
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

model_list=[logistic_model,mlp_model,knn_model,nb_model,svm_model,tree_model,
            bagging_model,adaboost_model,gradient_model, rf_model,xgb_model,lgb_model]

for i in model_list:
    print(,": ", cross_val_score(i, test_x, test_y, cv=StratifiedKFold(n_splits=3, random_state=777),scoring='accuracy'))

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(train_y, svm_model.predict(train_x))
confusion_matrix(test_y, svm_model.predict(test_x))
print(classification_report(test_y, svm_model.predict(test_x), target_names=['class 0', 'class 1', 'class2']))

























loan=pd.DataFrame( pd.read_csv('/home/keti/DataAnalysis/Tutorials/Multiple_Classification/Loan/Loan payments data.csv'))\
            .assign(effective_date=lambda x:pd.to_datetime(x['effective_date']),
                    due_date=lambda x:pd.to_datetime(x['due_date']),
                    paid_off_time=lambda x:pd.to_datetime(x['paid_off_time']),
                    past_due_days= lambda x:x['past_due_days'].replace(np.nan,0),
                    Gender=lambda x: x[['Gender']].applymap({'female':1, 'male':0}.get),
                    loan_status=lambda x: x[['loan_status']].applymap({'PAIDOFF':0, 'COLLECTION_PAIDOFF':1,'COLLECTION':2}.get), 
                    education=lambda x: pd.get_dummies(x[['education']]))\
                     .drop(['Loan_ID'], axis=1)


# voting model=====================================================================================================
from sklearn.ensemble import VotingClassifier

# voting, hard: 직접 투표 , 결과에서 가장 많이 나온 결과를 채택, 디폴트  soft: 간접 투표 ,  각 클래스 확률을 합한 것들 중 가장 큰 것를 채택
# weights : array-like, shape = [n_classifiers], optional (default=`None`)  ex) weights=[2,1,1]    
voting_model1 = VotingClassifier(estimators=[('lr', LogisticRegression()),
                                             ('svm', svm.SVC()),
                                             ('knn', KNeighborsClassifier()),
                                             ('tree', tree.DecisionTreeClassifier()),
                                             ('xgboost', xgboost.XGBClassifier()) ],
                                 voting='hard', weights=None)

voting_model2 = VotingClassifier(estimators=[('lr', LogisticRegression()),
                                             ('svm', svm.SVC( probability=True)),
                                             ('knn', KNeighborsClassifier()),
                                             ('tree', tree.DecisionTreeClassifier()),
                                             ('xgboost', xgboost.XGBClassifier()) ],
                                 voting='soft', weights=None)

voting_model1.fit(train_x, train_y)
voting_model2.fit(train_x, train_y)

##########모델 검증

confusion_matrix(test_y, voting_model1.predict(test_x))
confusion_matrix(test_y, voting_model2.predict(test_x))
print(classification_report(test_y, voting_model1.predict(test_x), target_names=['class 0', 'class 1', 'class2']))
print(classification_report(test_y, voting_model2.predict(test_x), target_names=['class 0', 'class 1', 'class2']))

print(voting_model1.score(test_x, test_y)) 
print(voting_model2.score(test_x, test_y)) 





# keras model=================================================================================================================

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 라벨링 전환
Y_train = np_utils.to_categorical(train_y)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(test_y)

# 2. 모델 구성하기
model1 = Sequential()
model1.add(Dense(units=3, input_dim=6, activation='softmax'))

model2 = Sequential()
model2.add(Dense(64, input_dim=6, activation='relu'))
model2.add(Dense(10, activation='softmax'))

# 3. 모델 엮기
model1.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
hist = model1.fit(train_x, train_y, epochs=1000, batch_size=10, validation_data=(X_val, Y_val))

# 5. 모델 학습 과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()



