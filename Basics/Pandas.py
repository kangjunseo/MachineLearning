#!/usr/bin/env python
# coding: utf-8


# ## Pandas

# In[1]:


import pandas as pd
import os


# In[2]:


os.getcwd()


# In[7]:


titanic_df=pd.read_csv('/Users/kangjunseo/python programming/파이썬 머신러닝 완벽 가이드/titanic_train.csv')
titanic_df.head(3)


# In[8]:


print(type(titanic_df))
print(titanic_df.shape)


# In[9]:


titanic_df.info() 


# In[10]:


titanic_df.describe()


# In[11]:


print(titanic_df['Pclass'].value_counts())


# In[12]:


titanic_df['Pclass'].head()


# ### DataFrame 변환

# In[13]:


import numpy as np


# In[17]:


#1d array to DataFrame

col_name1=['col1']
df_list=pd.DataFrame([1,2,3],columns=col_name1)
print(df_list)

arr=np.array([1,2,3])
df_arr=pd.DataFrame(arr,columns=col_name1)
print(df_arr)


# In[18]:


#2d array to DataFrame

col_name2=['col1','col2','col3']

df_list=pd.DataFrame([[1,2,3],
                      [11,12,13]],columns=col_name2)
print(df_list)

arr=np.array([[1,2,3],
              [11,12,13]])
df_arr=pd.DataFrame(arr,columns=col_name2)
print(df_arr)


# In[20]:


#Dictionary to DataFrame
dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print(df_dict)


# #### Dataframe to Others

# In[85]:


#ndarray
arr=df_dict.values
print(arr)

#list
_list=df_dict.values.tolist()
print(_list)

#dictionary
dict=df_dict.to_dict()
print(dict)


# ### DataFrame의 Column Data Set 생성 및 수정

# #### 새로운 칼럼 추가

# In[25]:


titanic_df['Age_0']=0 
titanic_df.head(3)


# In[27]:


titanic_df['Age_by_10']=titanic_df['Age']*10
titanic_df['Family_No']=titanic_df['SibSp']+titanic_df['Parch']+1
titanic_df.head(3)


# In[28]:


titanic_df['Age_by_10']=titanic_df['Age_by_10']+100
titanic_df.head(3)


# ### DataFrame 데이터 삭제

# #### inplace = False 인 경우

# In[32]:


#axis0=row, axis1=col
titanic_drop_df=titanic_df.drop('Age_0',axis=1)
titanic_drop_df.head()


# #### inplace = True 인 경우

# In[33]:


drop_result=titanic_df.drop(['Age_0','Age_by_10','Family_No'],axis=1,inplace=True)
print(drop_result) #inplace=True 이어서 반환값이 None
titanic_df.head(3)


# In[34]:


pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',15)
print(titanic_df.head(3))
titanic_df.drop([0,1,2],axis=0,inplace=True)
print(titanic_df.head(3))


# ### Index 객체

# In[37]:


titanic_df=pd.read_csv('titanic_train.csv')
indexes = titanic_df.index
print(indexes)
print(indexes.values) #1d array


# In[39]:


print(type(indexes.values))
print(indexes[:5].values)
print(indexes.values[:5])


# In[40]:


#한 번 생성된 DataFrame/Series의 Index 객체는 immutable
indexes[0]=5


# series 객체에 연산 함수 --> `Index`는 연산에서 제외

# In[42]:


series_fair = titanic_df['Fare']
print(series_fair.max() , series_fair.sum() ,sum(series_fair))
(series_fair+3).head(3)


# In[44]:


#인덱스가 연속된 int 숫자형 데이터가 아닐 경우 사용
titanic_reset_df=titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)


# In[45]:


value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print(type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False) #drop=True 시 index가 column으로 추가되지 않고 삭제됨
print(new_value_counts)
print(type(new_value_counts))


# ### Data Selection & Filtering

# #### DataFrame의 [] 연산자

# In[46]:


print(titanic_df['Pclass'].head(3)) #가능
print(titanic_df[0]) #불가능


# In[47]:


titanic_df[0:2] #pandas의 Index 형태로 변환가능, 하지만 사용하지 않는 것이 좋음


# In[48]:


titanic_df[titanic_df['Pclass']==3].head(3) #boolean indexing


# #### iloc[] - 위치 기반 Indexing

# In[50]:


data={'Name':['Chulmin','Eunkyung','Jinwoong','Soobeom'],
     'Year':[2011,2016,2015,2015],
     'Gender':['Male','Female','Male','Male']
     }
data_df=pd.DataFrame(data,index=['one','two','three','four'])
data_df


# In[51]:


data_df.iloc[0,0]


# In[52]:


data_df.iloc[0,'Name'] #명칭 입력시 오류
data_df.iloc['one,0']


# In[57]:


data_df_reset=data_df.reset_index()
data_df_reset=data_df_reset.rename(columns={'index':'old_index'})
data_df_reset.index = data_df_reset.index+1
print(data_df_reset)
data_df_reset.iloc[0,1] #index 값과 헷갈리지 않음


# #### loc[] - 명칭 기반 Indexing

# In[53]:


data_df.loc['one','Name']


# In[58]:


data_df_reset.loc[1,'Name'] #이 경우에는 인덱스 값으로 1이 존재하므로 가능


# In[59]:


print('위치 기반 iloc slicing\n', data_df.iloc[0:1,0],'\n') # 0부터 1-1까지
print('명칭 기반 loc slicing\n', data_df.loc['one':'two','Name']) #'one'부터 'two'까지 ('two'-1이 불가능하므로)


# In[60]:


print(data_df_reset.loc[1:2,'Name'])


# #### Boolean Indexing

# In[61]:


#Using []
titanic_df = pd.read_csv('titanic_train.csv')
titanic_boolean = titanic_df[titanic_df['Age']>60]
print(type(titanic_boolean))
titanic_boolean


# In[62]:


titanic_df[titanic_df['Age']>60][['Name','Age']].head(3)


# In[63]:


#Using loc[]
titanic_df.loc[titanic_df['Age']>60,['Name','Age']].head(3)


# In[65]:


# and = &, or = |, not = ~ 
# 60세 이상, 1등급, 여성
titanic_df[(titanic_df['Age']>60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')]
# 각 조건연산자 사이 괄호 필수


# In[66]:


IsOld = titanic_df['Age']>60
IsFirst = titanic_df['Pclass']==1
IsWoman = titanic_df['Sex']=='female'
titanic_df[IsOld & IsFirst & IsWoman]


# ### Sort, Aggregation, GroupBy

# #### sort_values()

# In[67]:


# main args - by, ascending, inplace
titanic_sorted = titanic_df.sort_values(by=['Name'])
titanic_sorted.head(3)


# In[68]:


titanic_sorted = titanic_df.sort_values(by=['Pclass','Name'],ascending=False)
titanic_sorted.head(3)


# #### Aggregation

# In[69]:


#min(), max(), sum(), count()
titanic_df.count()


# In[70]:


titanic_df[['Age','Fare']].mean()


# #### groupby()

# In[71]:


titanic_groupby = titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))


# In[72]:


titanic_groupby = titanic_groupby.count()
titanic_groupby


# In[73]:


titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId','Survived']].count()
titanic_groupby


# In[74]:


#aggregation 적용
titanic_df.groupby('Pclass')['Age'].agg([max, min])


# In[75]:


#각각 다른 agg 적용
agg_format = {'Age':'max','SibSp':'sum','Fare':'mean'}
titanic_df.groupby('Pclass').agg(agg_format)


# ### 결손 데이터 처리

# NULL 데이터는 NaN으로 나타남
# 머신러닝 알고리즘은 이 값을 처리하지 않으므로 대체해야함

# #### isna() - 결손 데이터 여부 확인

# In[76]:


titanic_df.isna().head(3)


# In[77]:


titanic_df.isna().sum()


# #### fillna() - 결손 데이터 대체

# In[78]:


titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')
titanic_df.head()


# In[79]:


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


# ### apply lambda

# In[89]:


arr=np.arange(10)
sq_arr = map(lambda x : x**2,arr)
print(list(sq_arr))


# In[90]:


titanic_df['Name_len']=titanic_df['Name'].apply(lambda x :len(x))
titanic_df[['Name','Name_len']].head(3)


# In[93]:


titanic_df['Child_Adult']=titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else 'Adult')
titanic_df[['Age','Child_Adult']].head(8)


# In[95]:


titanic_df['Age_cat']=titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else('Adult' if x<=60 else 'Elder'))
titanic_df['Age_cat'].value_counts()


# In[96]:


def get_cat(age):
    cat =''
    if age<=5: cat='Baby'
    elif age<=12: cat='Child'
    elif age<=18: cat='Teen'
    elif age<=25: cat='학식'
    elif age<=35: cat='Adult'
    else : cat='Grand Grand Adult'
        
    return cat

titanic_df['Age_cat']= titanic_df['Age'].apply(lambda x : get_cat(x))
titanic_df[['Age','Age_cat']].head()

