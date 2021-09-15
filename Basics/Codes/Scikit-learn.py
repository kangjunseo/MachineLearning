#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn

print(sklearn.__version__)


# ## First ML : Iris Classification

# In[3]:


from sklearn.datasets import load_iris #붓꽃 데이터 세트
from sklearn.tree import DecisionTreeClassifier #의사 결정 트리 알고리즘
from sklearn.model_selection import train_test_split #학습 데이터와 테스터 데이터 분리


# In[5]:


import pandas as pd
iris = load_iris()
iris_data = iris.data  #feature 데이터
iris_label = iris.target  #레이블 데이터 (0 = Setosa, 1 = versicolor, 2 = virginica)

print(iris_label)
print(iris.target_names)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.head(3)


# In[6]:


# 테스트 데이터 20%, 학습 데이터 80%로 분할 / X : feature, Y : label
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)


# In[7]:


dt_clf = DecisionTreeClassifier(random_state=11)

#학습
dt_clf.fit(X_train,y_train)


# In[8]:


#학습 완료 후 예측 수행
pred = dt_clf.predict(X_test)


# In[9]:


#정확도 측정
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))


# ## Scikit-Learn의 기반 FrameWork

# 

# ### Estimator, fit(), predict()

# #### Supervised Learning
# Esimator Class - Classifier/ Regressor  
#     ->fit(), predict()

# #### Unsupervised Learning
# Dimensionality Reduction,Clustering, Feature Extraction  
# ->fit(), transform();

# 

# ### 내장된 예제 데이터 세트 활용

# data - feature의 data set  
# target - 분류 시 label, 회귀 시 숫자 결괏값 data set  
# target_names - 개별 label의 이름  
# feature_names - feature의 이름  
# DESCR - data set, feature의 description

# In[11]:


from sklearn.datasets import load_iris

iris_data=load_iris()
print(type(iris_data)) #Bunch는 Python dictionary와 유사


# In[12]:


keys = iris_data.keys()
print(keys)


# In[13]:


print('feature_name type : ',type(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names type : ',type(iris_data.target_names))
print(iris_data.target_names)

print('\n data type : ',type(iris_data.data) )
print(iris_data.data)

print('\n target type : ',type(iris_data.target))
print(iris_data.target)


# ## Model Selection

# model_selection 모듈 - train/test split, 교차 검증 분할 및 평가, Estimator의 hyper parameter 튜닝

# ### train_test_split()

# 먼저 테스트 데이터 없이 학습 데이터 세트로만 학습하고 예측(wrong case)

# In[16]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data #train data를 분리없이 전체 data set으로 설정
train_label = iris.target
dt_clf.fit(train_data, train_label)

pred = dt_clf.predict(train_data)
print(accuracy_score(train_label,pred))


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.3,random_state=121)


# In[19]:


dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))


# ### 교차 검증

# Overfitting 방지를 위해 train data를 다시 train/test로 나눔

# #### K 폴드 교차 검증
# K개 데이터 폴드 세트 생성 -> K번만큼 각 폴트 세트에 학습/검증 반복

# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5세트로 분리하는 KFold 객체와 각 세트별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits=5)
cv_accuracy=[]
print(features.shape[0])


# In[23]:


n_iter = 0

#kfold.split()을 통해 각 폴드 별 index를 array로 반환
for train_index, test_index in kfold.split(features):
    #반환된 index를 이용해 데이터 분리
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    
    #train & predict
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter +=1
    
    #각 iteration마다 accuracy 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습데이터 크기: {2}, 검증 데이터 크기: {3}'.format(n_iter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter, test_index))
    cv_accuracy.append(accuracy)
    
print('\n## 평균 검증 정확도:',np.mean(cv_accuracy))


# #### Stratified K 폴드
# 불균형한 분포도를 가진 label data collection을 위한 K 폴드  
# 원본 data의 label 분포를 먼저 고려한 뒤, 이 분포와 동일하게 train/test data set 설정

# In[25]:


#K폴드의 한계

import pandas as pd
iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()


# In[26]:


kfold = KFold(n_splits=3)
n_iter=0
for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())


# 위와 같이 진행할 경우 train label data와 test label data가 완전히 다르므로 정확도가 0이 되어버림

# In[27]:


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_index, test_index in skf.split(iris_df,iris_df['label']):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())


# In[28]:


dt_clf=DecisionTreeClassifier(random_state=156)

skfold=StratifiedKFold(n_splits=3)
n_iter=0
cv_accuracy=[]

#split() 호출 시 label data set도 추가 입력
for train_index, test_index in skfold.split(features,label):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    #반복마다 정확도 측정
    n_iter+=1
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'.format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
    print('\n## 교차 검증별 정확도:', np.round(cv_accuracy,4))
    print('## 평균 검증 정확도: ',np.mean(cv_accuracy))


# #### 보다 간편한 교차 검증 - cross_val_score()
# KFold 과정 : 
# 
#             1. 폴드 세트 설정  
#             2. for loop로 인덱스 추출  
#             3. 반복 학습

# cross_val_score() : 위 과정을 한번에 진행  
# estimator : classifier or regressor  
# X : feature data set, y : label data set  
# scoring : 예측 성능 평가 지표, cv : 교차 검증 폴드 수

# 분류 -> Stratified K fold / 회귀 -> K fold

# In[29]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

#성능 지표 : accuracy, 교차 검증 세트 수 : 3
scores = cross_val_score(dt_clf, data, label, scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores,4))
print('평균 검증 정확도:',np.round(np.mean(scores),4))


# 

# ### GridSearchCV - 교차 검증 + Optimal Hyper Parameter 튜닝

# Hyper Parameter의 조정을 통해 알고리즘의 예측 성능 개선 가능

# In[33]:


grid_parameters = {'max_depth':[1,2,3],
                   'min_samples_split':[2,3]
                  }


# In[34]:


from sklearn.model_selection import GridSearchCV

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,test_size=0.2,random_state=121)
dtree = DecisionTreeClassifier()
parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}


# In[36]:


grid_dtree = GridSearchCV(dtree, param_grid=parameters,cv=3, refit=True)
grid_dtree.fit(X_train,y_train)
scores_df=pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]


# In[37]:


print('Best Case :', grid_dtree.best_params_,grid_dtree.best_score_)


# In[38]:


#GridSearchCV의 refit로 학습된 estimator 반환 (refit=True(Default)일 때 활성화)
estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)
print(accuracy_score(y_test,pred))


# 

# ## 데이터 전처리

# 결손값(NaN, Null) 불허 -> 불필요한 경우 해당 feature drop    
# 문자열 변환 - 카테고리형 / 텍스트형 feature

# ### Data Encoding

# Label encoding - 카테고리 feature -> 코드형 숫자

# #### Label encoding

# In[39]:


from sklearn.preprocessing import LabelEncoder

items=['TV','냉장고', '전자레인지', '컴퓨터', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print(labels)


# In[40]:


print(encoder.classes_)


# In[41]:


encoder.inverse_transform([4,5,2,0,1,1,3,3])


# Label Encoding은 값을 숫자로 반환하므로 회귀와 같은 ML 알고리즘에는 적용 x (트리 계열은 가능)

# 

# #### One-Hot Encoding

# In[45]:


from sklearn.preprocessing import OneHotEncoder
import numpy as np

items = ['TV','냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

#숫자 값으로 변환하기 위해 LabelEncoder 먼저 사용
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items).reshape(-1,1) #2d array

#Apply One-Hot Encoding
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print(oh_labels.toarray())
print(oh_labels.shape)


# In[46]:


#One-Hot Encoding in Pandas
import pandas as pd

df = pd.DataFrame({'item':['TV','냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
pd.get_dummies(df)


# 

# ### Feature scaling and Normalization

# #### StandardScaler

# In[51]:


iris_df.drop(['label'],axis=1,inplace=True)
print(iris_df.mean())
print('\n',iris_df.var())


# In[53]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
iris_df_scaled = pd.DataFrame(data = iris_scaled,columns=iris.feature_names)

print(iris_df_scaled.mean())
print('\n',iris_df_scaled.var())


# #### MinMaxScaler

# In[55]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(iris_df)
iris_scaled=scaler.transform(iris_df)
iris_df_scaled=pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print('min : ',iris_df_scaled.min())
print('\nmax : ',iris_df_scaled.max())


# ### fit_transform() 주의사항

# In[67]:


from sklearn.preprocessing import MinMaxScaler

train_array = np.arange(0,11).reshape(-1,1)
test_array = np.arange(0,6).reshape(-1,1)


# In[69]:


scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)

print('Original :',np.round(train_array.reshape(-1),2))
print('Scaled :',np.round(train_scaled.reshape(-1),2))


# In[70]:


scaler.fit(test_array)
test_scaled = scaler.transform(test_array)
print('Original :',np.round(test_array.reshape(-1),2))
print('Scaled :',np.round(test_scaled.reshape(-1),2))


# 위의 결과를 보면 1은 0.1로 scale 되야 하는데, fit()을 한번더 진행해서 0.2로 scale 되었음  
#   
# 위 상황과 같이 fit_transform()을 테스트 데이터에서는 사용하면 안됨

# 

# In[71]:


scaler=MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print('Original :',np.round(train_array.reshape(-1),2))
print('Scaled :',np.round(train_scaled.reshape(-1),2))

#이번에는 fit을 다시 호출하지 않음
test_scaled = scaler.transform(test_array)
print('Original :',np.round(test_array.reshape(-1),2))
print('Scaled :',np.round(test_scaled.reshape(-1),2))


# 가능하다면 전체 데이터의 스케일링 변환을 먼저 적용한 뒤 train test를 분리

# In[ ]:




