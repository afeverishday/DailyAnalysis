#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:14:55 2020

@author: Jay
"""
## ( multi-layer perceptron 모델)
import gc
gc.collect()
import warnings
warnings.filterwarnings('ignore') # 경고 출력하지 않음

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rawdata = pd.read_csv('/home/keti/DataAnalysis/Tutorials/Multiple_Classification/phenotype/phenotype.txt', sep='	')\
            .assign(Phenotype=lambda x: x[['Phenotype']].applymap({ 'desert': 0,'excluded': 1, 'inflamed':2}.get) ).drop(['SampleID'], axis=1)

# EDA=====================================================
rawdata.columns
rawdata.head()
eda=rawdata.describe().round(3)
rawdata.info()
rawdata.isnull().sum()
(rawdata.isnull().sum()/max(rawdata.count())).round(2)

import collections
from natsort import natsorted
collections.Counter(natsorted(rawdata['Phenotype']))

# columns name sort
categorical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes == "object"]))
numerical_feature = list( set([ col for col in rawdata.columns if rawdata[col].dtypes in(['float64', 'int64']) ]))
minmax_feature = list( set(numerical_feature)-set(['Phenotype']))
target_feature = list(['Phenotype'])
#dummy_feature=list(set(categorical_feature))

from pytictoc import TicToc
tictoc = TicToc() #create instance of class
#tictoc.tic() #Time elapsed since t.tic()

# scaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
minmaxscaler = MinMaxScaler().fit(rawdata[minmax_feature])
train_minmax = pd.DataFrame(minmaxscaler.transform(rawdata[minmax_feature]),
                                columns= minmax_feature, index=list(rawdata.index.values))
#train_minmax = rawdata[minmax_feature]
train_target = pd.get_dummies(rawdata[target_feature], prefix='Phenotype', drop_first=False)
#train_dummy=pd.get_dummies(train_x[categorical_feature], prefix=categorical_feature, drop_first=True)
       
train_all=pd.concat([train_minmax, train_target], axis=1)# columns bind

#Correlattion=========================================
for col in numerical_feature:
    sns.distplot(rawdata.loc[rawdata[col].notnull(), col])
    plt.title(col)
    plt.show()

sns.pairplot(rawdata[list(numerical_feature)], hue='Phenotype', 
             x_vars=numerical_feature, y_vars=numerical_feature)
plt.show()

# corr and mic graph
corr= train_all.corr(method='pearson')\
        .loc[target_feature].round(3)
       
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est="mic_approx")

plt.figure(figsize=(20, 20))
plt.subplots_adjust(wspace=None, hspace=1)
i=0
for j in train_minmax.columns:
    i=i+1
    print(i,j)
    plt.subplot(820+i)
    plt.scatter(train_all[j], train_all['Phenotype'], c=train_all.Phenotype)
    mine.compute_score(train_all[j], train_all['Phenotype'])
    plt.title("MIC={0:0.3f}, corr={0:0.3f}".format(mine.mic(),corr.iloc[0,i-1] ))
    plt.xlabel(j, fontsize=12)
    plt.ylabel('Phenotype', fontsize=12)


byphenotype=train_all.groupby('Phenotype').agg(['size', 'mean', 'std', 'min', 'max'])
for i in train_minmax.columns:
    print(i)
    print(byphenotype[i])


train_x= train_minmax
train_y= train_target

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_x)
pca.explained_variance_ratio_
pca.components_
pca.n_features_
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, train_y], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0,1,2]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Phenotype'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

corr= finalDf.corr(method='pearson')\
        .loc[target_feature].round(3)

train_x=  pd.concat([train_minmax,principalDf], axis=1)



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
param_grid = {'max_iter':np.arange(5,100,5), 'alpha':np.arange(0.1,1,0.1),'hidden_layer_sizes':np.arange(10,200,10),'activation':['identity', 'logistic', 'tanh', 'relu']}
mlp_model = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777),refit=True).fit(train_x, train_y)
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
param_grid = {}
nb_model = GridSearchCV( GaussianNB(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='accuracy',refit=True).fit(train_x, train_y)
print(nb_model.best_score_)

from sklearn.tree import DecisionTreeClassifier
   # criterion : 'gini', 'entropy'(default=”gini”) split 할 특성 선택 알고리즘, 어떤 기준으로 정보 획득량을 계산해 가지를 분리 할 것인지
   # max_depth : int(default=None) 얼마나 깊게 트리를 만들어 갈거냐   None 이면 최대한 깊게 (불순도 혹은 복잡도가 0일 때까지)   클수록 정확하게 (과대적합),   작을수록 가지치기 (과대적합 방지)
   # max_leaf_nodes : int(default=None) 최대 몇개 잎 노드가 만들어 질때 까지  split(하위 (잎) 노드로 분리) 할 것이냐   클수록 정확하게 (과대적합),   작을수록 가지치기 (과대적합 방지)
   # min_samples_split : float(default=2) 샘플이 최소한 몇개 이상이어야  split(하위 (잎) 노드로 분리) 할거냐
   #  int일 경우 주어진 값을 그대로 사용, float일 경우 0에서 1사이의 값을 줄 수 있으며 전체 데이터 수*min_sample_split의 값을 사용 # 클수록 가지치기 (과대적합 방지),   작을수록 정확하게 (과대적합)   
   # min_samples_leaf : int, float, optional (default=1) (잎) 노드가 되려면 가지고 있어야할 최소 샘플 수     #  클수록 가지치기 (과대적합 방지),  작을수록 정확하게 (과대적합)
param_grid = {'criterion' :['gini','entropy'],'max_features':np.arange(1,9,1), 'max_depth':np.arange(1,15,1)}
tree_model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777),refit=True).fit(train_x, train_y)
print(tree_model.best_params_)
print(tree_model.best_score_)
tree_model = DecisionTreeClassifier(criterion= 'gini', max_depth=9, max_features= 6).fit(train_x, train_y)
tree_model.feature_importances_
n_feature = train_x.shape[1]
index = np.arange(n_feature)
plt.barh(index, tree_model.feature_importances_, align='center')
plt.yticks(index, train_x.columns)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()




from sklearn.svm import SVC
#    kernel : linear, poly,rbf
#    C : float, optional (default=1.0) 클수록 정확하게 (마진이 작아짐, 과대적합),  alpha (가중치 규제) 의 역수
#    gamma : float, optional (default=’auto’) 클수록 정확하게 (경사가 급해짐, 과대적합)   비선형 (kernel=‘rbf’, kernel=‘poly’)에서만 이 옵션 사용 
param_grid = {'kernel':['linear','rbf','poly'], 'C':np.arange(1,10,1)}
svm_model = GridSearchCV(SVC(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777),refit=True).fit(train_x, train_y)
print(svm_model.best_params_)
print(svm_model.best_score_)

from sklearn.ensemble import VotingClassifier
# voting, hard: 직접 투표 , 결과에서 가장 많이 나온 결과를 채택, 디폴트  soft: 간접 투표 ,  각 클래스 확률을 합한 것들 중 가장 큰 것를 채택
# weights : array-like, shape = [n_classifiers], optional (default=`None`)  ex) weights=[2,1,1]    
voting_modelset1 = VotingClassifier(estimators=[('lr', LogisticRegression()), 
                                             ('knn', KNeighborsClassifier()),
                                             ('tree', DecisionTreeClassifier()),
                                             ('nb', GaussianNB()),
                                             ('svc', SVC())],
                                              voting='hard', weights=None).fit(train_x, train_y)

param_grid = {}
voting_model1 = GridSearchCV( voting_modelset1, param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='accuracy',refit=True).fit(train_x, train_y)
print(voting_model1.best_score_)

voting_modelset2 = VotingClassifier(estimators=[('lr', LogisticRegression()), 
                                             ('knn', KNeighborsClassifier()),
                                             ('tree', DecisionTreeClassifier()),
                                             ('nb', GaussianNB()),
                                             ('svc', SVC(probability=True))],
                                              voting='soft', weights=None).fit(train_x, train_y)
param_grid = {}
voting_model2 = GridSearchCV( voting_modelset2, param_grid=param_grid, cv=KFold(n_splits=5, random_state=777), scoring='accuracy',refit=True).fit(train_x, train_y)
print(voting_model2.best_score_)


from sklearn.ensemble import BaggingClassifier
#    base_estimator: base estimator to fit
#    n_estimators: the number of base estimators 
#    bootstrap : (default=True) 중복 할당 여부  True 베깅, S  False 페이스팅
param_grid = {'base_estimator':[DecisionTreeClassifier(),LogisticRegression(),KNeighborsClassifier()], 'n_estimators':np.arange(2,20,2),'bootstrap':[True, 'S',False]}
bagging_model = GridSearchCV(BaggingClassifier(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777),refit=True).fit(train_x, train_y)
print(bagging_model.best_params_)
print(bagging_model.best_score_)

from sklearn.ensemble import RandomForestClassifier
#    n_estimators : integer, optional (default=10) The number of trees in the forest.
#    bootstrap : boolean, optional (default=True) True 베깅,   False 페이스팅
#    criterion : string, optional (default=”mse”) 'mse' (평균제곱오차),   'friedman_mse', 'mae'
#    min_impurity_split=None, min_samples_leaf=1, mins_samples_split=2, min_sample_leat=1,
#    min_samples_split=2, min_waight_fraction_leaf=0.0, presort=false, random_state=0, splittter='best'
param_grid = {'n_estimators':np.arange(2,10,1),'bootstrap':[True,False],'max_features':np.arange(1,9,1), 'max_depth':np.arange(1,10,1)}
rf_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=KFold(n_splits=5, random_state=777),refit=True).fit(train_x, train_y)
print(rf_model.best_params_)
print(rf_model.best_score_)
rf_model = RandomForestClassifier(bootstrap= False, max_depth= 6, max_features= 2, n_estimators= 7 ).fit(train_x, train_y)
rf_model.feature_importances_

n_feature = train_x.shape[1]
index = np.arange(n_feature)
plt.barh(index, rf_model.feature_importances_, align='center')
plt.yticks(index, train_x.columns)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()



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
from sklearn.metrics import recall_score,roc_auc_score, f1_score

model_list=[logistic_model,knn_model,nb_model,svm_model,tree_model, voting_model1,voting_model2,
            bagging_model,adaboost_model,gradient_model, rf_model]

modelname_list=['logistic_model','knn_model','nb_model','svm_model', 'tree_model', 'voting_model1', 'voting_model2',
            'bagging_model','adaboost_model','gradient_model','rf_model']

from sklearn.metrics import confusion_matrix,classification_report
j=0
for i in model_list :
    print( modelname_list[j])
    j=j+1
    print("accuracy_score: ", cross_val_score(i, train_minmax, train_target, cv=StratifiedKFold(n_splits=5, random_state=777),scoring='accuracy'))
    print("recall_score: ", cross_val_score(i, train_minmax, train_target, cv=StratifiedKFold(n_splits=5, random_state=777),scoring='recall_macro'))
    print("f1_score: ", cross_val_score(i, train_minmax, train_target, cv=StratifiedKFold(n_splits=5, random_state=777),scoring='f1_weighted'))
    print(confusion_matrix(train_y, i.predict(train_x)))
    print(classification_report(train_y, i.predict(train_x), target_names=['class 0', 'class 1', 'class 2']))




confusion_matrix(train_target, svm_model.predict(train_minmax))
print(classification_report(train_target, svm_model.predict(train_minmax), target_names=['class 0', 'class 1', 'class 2']))
