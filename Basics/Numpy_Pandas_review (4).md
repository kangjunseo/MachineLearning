```python
import numpy as np
```

## Numpy

### 배열 생성 및 재구성
`np.array()` : 인자를 ndarray로 `return`    
`shape` : ndarray의 dimension과 size를 tuple로 `return`


```python
array = np.array([[1,2,3],[4,5,6]])
print('array type:', type(array))
print('array 형태:', array.shape)
```

    array type: <class 'numpy.ndarray'>
    array 형태: (2, 3)


ndarray의 data type은 모두 같아야함.  
---> 더 큰 data type으로 교체


```python
list1 = [1,2,'test']
array1=np.array(list1)
print(array1,array1.dtype)

list2 =[1,2,3.0]
array2=np.array(list2)
print(array2,array2.dtype)
```

    ['1' '2' 'test'] <U21
    [1. 2. 3.] float64


ML Algorithm에서는 대용량의 데이터를 다루기 때문에 데이터의 용량을 줄이기도 함  
--->`astype()` 함수 이용  
하지만 부적절하게 변경하면 데이터 손실 가능성 있음


```python
array_int=np.array([1,2,3])
array_float=array_int.astype('float64')
print(array_float,array_float.dtype)

array_float1=np.array([1.1,2.2,3.3])
print(array_float1,array_float.dtype)

array_int1=array_float1.astype('int32')
print(array_int1,array_int1.dtype)

```

    [1. 2. 3.] float64
    [1.1 2.2 3.3] float64
    [1 2 3] int32


### ndarry 초기화 - arange, zeros, ones


```python
print(np.arange(10))
print(np.zeros((3,2),dtype='int32'))
print(np.ones((3,2)))
```

    [0 1 2 3 4 5 6 7 8 9]
    [[0 0]
     [0 0]
     [0 0]]
    [[1. 1.]
     [1. 1.]
     [1. 1.]]


### reshape()
인자로 변환하고 싶은 사이즈의 `tuple`을 전달.  
하지만 불가능한 사이즈 전달시 오류가 발생함.


```python
array1=np.arange(10)
print(array1)
print(array1.reshape(2,5))
print(array1.reshape(5,2))
```

    [0 1 2 3 4 5 6 7 8 9]
    [[0 1 2 3 4]
     [5 6 7 8 9]]
    [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]


-1을 인자로 사용 --> 자동으로 알맞은 사이즈  
하지만, 역시 불가능하거나 -1이 양쪽에 오는 경우에는 오류를 발생시킴.


```python
arr=np.arange(10)
print(arr,'\n',arr.reshape(-1,5),'\n',arr.reshape(5,-1))
```

    [0 1 2 3 4 5 6 7 8 9] 
     [[0 1 2 3 4]
     [5 6 7 8 9]] 
     [[0 1]
     [2 3]
     [4 5]
     [6 7]
     [8 9]]


`reshape(-1,1)` 활용하여 2d array화 시키기  
`tolost()` : ndarray to list


```python
arr3d=np.arange(8).reshape((2,2,2))
print(arr3d)

#3d to 2d using reshape(-1,1)
print(arr3d.reshape(-1,1).tolist())

arr1d=np.arange(8)
print(arr1d)

#1d to 2d using reshape(-1,1)
print(arr1d.reshape(-1,1).tolist())
```

    [[[0 1]
      [2 3]]
    
     [[4 5]
      [6 7]]]
    [[0], [1], [2], [3], [4], [5], [6], [7]]
    [0 1 2 3 4 5 6 7]
    [[0], [1], [2], [3], [4], [5], [6], [7]]


### Indexing  
<br/>

#### 특정 데이터만 추출 



```python
arr=np.arange(1,10)
print(arr)
print(arr[2],type(arr[2])) #3번째 값
print(arr[-2]) #맨 뒤에서 2번째

arr2=arr.reshape(3,3)
print(arr2)
print(arr2[1][1])
```

    [1 2 3 4 5 6 7 8 9]
    3 <class 'numpy.int64'>
    8
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    5


### Slicing


```python
arr=np.arange(1,10)
print(arr[:],arr[:3],arr[3:])

arr=arr.reshape(3,3)
print(arr[1:])
```

    [1 2 3 4 5 6 7 8 9] [1 2 3] [4 5 6 7 8 9]
    [[4 5 6]
     [7 8 9]]


### Fancy Indexing


```python
arr=np.arange(1,10).reshape(3,3)
print(arr[[0,1],2].tolist())
print(arr[[0,1],0:2])
```

    [3, 6]
    [[1 2]
     [4 5]]


### Boolean Indexing


```python
arr=np.arange(1,10)
arr2=arr[arr>5]
print(arr2)
print(arr>5)
print(arr[[False,False,False,False,False,True,True,True,True]])
print(arr[[5,6,7,8]])
```

    [6 7 8 9]
    [False False False False False  True  True  True  True]
    [6 7 8 9]
    [6 7 8 9]


### Sorting - sort()와 argsort()


```python
arr=np.array([3,1,9,5])
print(np.sort(arr))
print(arr)
print(arr.sort())
print(arr)
```

    [1 3 5 9]
    [3 1 9 5]
    None
    [1 3 5 9]



```python
arr=np.sort(np.arange(1,10))[::-1]
print(arr)
```

    [9 8 7 6 5 4 3 2 1]



```python
arr=np.array([[8,12],
             [7,1]])
print(np.sort(arr)) #axis=1, column을 한 덩이로 보면서 sort
print(np.sort(arr,0)) #axis=0, row를 한 덩이로 보면서 sort
```

    [[ 8 12]
     [ 1  7]]
    [[ 7  1]
     [ 8 12]]


#### 정렬된 행렬의 인덱스 반환


```python
arr=np.array([3,1,9,5])
print(np.argsort(arr))
print(np.argsort(arr)[::-1])
```

    [1 0 3 2]
    [2 3 0 1]



```python
name_arr=np.array(['John','Mike','Sarah','Kate','Samuel'])
score=np.array([78,95,84,98,88])
print(np.argsort(score))
print(name_arr[np.argsort(score)])
```

    [0 2 4 1 3]
    ['John' 'Sarah' 'Samuel' 'Mike' 'Kate']


### 선형대수 연산


```python
A=np.array(np.arange(1,7).reshape(2,3))
B=np.array(np.arange(7,13).reshape(3,2))
print(np.dot(A,B))
print(np.transpose(np.dot(A,B)))
```

    [[ 58  64]
     [139 154]]
    [[ 58 139]
     [ 64 154]]



```python

```

## Pandas


```python
import pandas as pd
import os
```


```python
os.getcwd()
```




    '/Users/kangjunseo/python programming/파이썬 머신러닝 완벽 가이드'




```python
titanic_df=pd.read_csv('/Users/kangjunseo/python programming/파이썬 머신러닝 완벽 가이드/titanic_train.csv')
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(type(titanic_df))
print(titanic_df.shape)
```

    <class 'pandas.core.frame.DataFrame'>
    (891, 12)



```python
titanic_df.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
titanic_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(titanic_df['Pclass'].value_counts())
```

    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64



```python
titanic_df['Pclass'].head()
```




    0    3
    1    1
    2    3
    3    1
    4    3
    Name: Pclass, dtype: int64



### DataFrame 변환


```python
import numpy as np
```


```python
#1d array to DataFrame

col_name1=['col1']
df_list=pd.DataFrame([1,2,3],columns=col_name1)
print(df_list)

arr=np.array([1,2,3])
df_arr=pd.DataFrame(arr,columns=col_name1)
print(df_arr)
```

       col1
    0     1
    1     2
    2     3
       col1
    0     1
    1     2
    2     3



```python
#2d array to DataFrame

col_name2=['col1','col2','col3']

df_list=pd.DataFrame([[1,2,3],
                      [11,12,13]],columns=col_name2)
print(df_list)

arr=np.array([[1,2,3],
              [11,12,13]])
df_arr=pd.DataFrame(arr,columns=col_name2)
print(df_arr)
```

       col1  col2  col3
    0     1     2     3
    1    11    12    13
       col1  col2  col3
    0     1     2     3
    1    11    12    13



```python
#Dictionary to DataFrame
dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print(df_dict)
```

       col1  col2  col3
    0     1     2     3
    1    11    22    33


#### Dataframe to Others


```python
#ndarray
arr=df_dict.values
print(arr)

#list
list=df_dict.values.tolist()
print(list)

#dictionary
dict=df_dict.to_dict()
print(dict)
```

    [[ 1  2  3]
     [11 22 33]]
    [[1, 2, 3], [11, 22, 33]]
    {'col1': {0: 1, 1: 11}, 'col2': {0: 2, 1: 22}, 'col3': {0: 3, 1: 33}}


### DataFrame의 Column Data Set 생성 및 수정

#### 새로운 칼럼 추가


```python
titanic_df['Age_0']=0 
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df['Age_by_10']=titanic_df['Age']*10
titanic_df['Family_No']=titanic_df['SibSp']+titanic_df['Parch']+1
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
      <th>Agw_by_10</th>
      <th>Family_No</th>
      <th>Age_by_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>220.0</td>
      <td>2</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>380.0</td>
      <td>2</td>
      <td>380.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>260.0</td>
      <td>1</td>
      <td>260.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df['Age_by_10']=titanic_df['Age_by_10']+100
titanic_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Age_0</th>
      <th>Agw_by_10</th>
      <th>Family_No</th>
      <th>Age_by_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>220.0</td>
      <td>2</td>
      <td>320.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>380.0</td>
      <td>2</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>260.0</td>
      <td>1</td>
      <td>360.0</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame 데이터 삭제

#### inplace = False 인 경우


```python
#axis0=row, axis1=col
titanic_drop_df=titanic_df.drop('Age_0',axis=1)
titanic_drop_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Family_No</th>
      <th>Age_by_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>2</td>
      <td>320.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>2</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>360.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>2</td>
      <td>450.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>450.0</td>
    </tr>
  </tbody>
</table>
</div>



#### inplace = True 인 경우


```python
drop_result=titanic_df.drop(['Age_0','Age_by_10','Family_No'],axis=1,inplace=True)
print(drop_result) #inplace=True 이어서 반환값이 None
titanic_df.head(3)
```

    None





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',15)
print(titanic_df.head(3))
titanic_df.drop([0,1,2],axis=0,inplace=True)
print(titanic_df.head(3))
```

       PassengerId  Survived  Pclass            Name     Sex   Age  SibSp  Parch          Ticket     Fare Cabin Embarked
    0            1         0       3  Braund, Mr....    male  22.0      1      0       A/5 21171   7.2500   NaN        S
    1            2         1       1  Cumings, Mr...  female  38.0      1      0        PC 17599  71.2833   C85        C
    2            3         1       3  Heikkinen, ...  female  26.0      0      0  STON/O2. 31...   7.9250   NaN        S
       PassengerId  Survived  Pclass            Name     Sex   Age  SibSp  Parch  Ticket     Fare Cabin Embarked
    3            4         1       1  Futrelle, M...  female  35.0      1      0  113803  53.1000  C123        S
    4            5         0       3  Allen, Mr. ...    male  35.0      0      0  373450   8.0500   NaN        S
    5            6         0       3  Moran, Mr. ...    male   NaN      0      0  330877   8.4583   NaN        Q


### Index 객체


```python
titanic_df=pd.read_csv('titanic_train.csv')
indexes = titanic_df.index
print(indexes)
print(indexes.values) #1d array
```

    RangeIndex(start=0, stop=891, step=1)
    [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
      18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
      36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
      54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
      72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
      90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
     108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
     126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
     144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
     162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
     180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
     198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
     216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
     234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
     252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269
     270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287
     288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305
     306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323
     324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341
     342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359
     360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377
     378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395
     396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413
     414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431
     432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449
     450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467
     468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485
     486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503
     504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521
     522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539
     540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557
     558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575
     576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593
     594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611
     612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629
     630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647
     648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665
     666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683
     684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701
     702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719
     720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737
     738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755
     756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773
     774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791
     792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809
     810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827
     828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845
     846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863
     864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881
     882 883 884 885 886 887 888 889 890]



```python
print(type(indexes.values))
print(indexes[:5].values)
print(indexes.values[:5])
```

    <class 'numpy.ndarray'>
    [0 1 2 3 4]
    [0 1 2 3 4]



```python
#한 번 생성된 DataFrame/Series의 Index 객체는 immutable
indexes[0]=5
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /var/folders/hg/s9q7g8q94b13_96p488f3z3m0000gn/T/ipykernel_81587/737481228.py in <module>
          1 #한 번 생성된 DataFrame/Series의 Index 객체는 immutable
    ----> 2 indexes[0]=5
    

    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexes/base.py in __setitem__(self, key, value)
       4583     @final
       4584     def __setitem__(self, key, value):
    -> 4585         raise TypeError("Index does not support mutable operations")
       4586 
       4587     def __getitem__(self, key):


    TypeError: Index does not support mutable operations


series 객체에 연산 함수 --> `Index`는 연산에서 제외


```python
series_fair = titanic_df['Fare']
print(series_fair.max() , series_fair.sum() ,sum(series_fair))
(series_fair+3).head(3)
```

    512.3292 28693.9493 28693.949299999967





    0    10.2500
    1    74.2833
    2    10.9250
    Name: Fare, dtype: float64




```python
#인덱스가 연속된 int 숫자형 데이터가 아닐 경우 사용
titanic_reset_df=titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mr...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, ...</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 31...</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print(type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False) #drop=True 시 index가 column으로 추가되지 않고 삭제됨
print(new_value_counts)
print(type(new_value_counts))
```

    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64
    <class 'pandas.core.series.Series'>
       index  Pclass
    0      3     491
    1      1     216
    2      2     184
    <class 'pandas.core.frame.DataFrame'>


### Data Selection & Filtering

#### DataFrame의 [] 연산자


```python
print(titanic_df['Pclass'].head(3)) #가능
print(titanic_df[0]) #불가능
```

    0    3
    1    1
    2    3
    Name: Pclass, dtype: int64



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3360             try:
    -> 3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 0

    
    The above exception was the direct cause of the following exception:


    KeyError                                  Traceback (most recent call last)

    /var/folders/hg/s9q7g8q94b13_96p488f3z3m0000gn/T/ipykernel_81587/28059505.py in <module>
          1 print(titanic_df['Pclass'].head(3)) #가능
    ----> 2 print(titanic_df[0]) #불가능
    

    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/frame.py in __getitem__(self, key)
       3453             if self.columns.nlevels > 1:
       3454                 return self._getitem_multilevel(key)
    -> 3455             indexer = self.columns.get_loc(key)
       3456             if is_integer(indexer):
       3457                 indexer = [indexer]


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       3361                 return self._engine.get_loc(casted_key)
       3362             except KeyError as err:
    -> 3363                 raise KeyError(key) from err
       3364 
       3365         if is_scalar(key) and isna(key) and not self.hasnans:


    KeyError: 0



```python
titanic_df[0:2] #pandas의 Index 형태로 변환가능, 하지만 사용하지 않는 것이 좋음
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mr...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_df[titanic_df['Pclass']==3].head(3) #boolean indexing
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr....</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, ...</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 31...</td>
      <td>7.925</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. ...</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.050</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### iloc[] - 위치 기반 Indexing


```python
data={'Name':['Chulmin','Eunkyung','Jinwoong','Soobeom'],
     'Year':[2011,2016,2015,2015],
     'Gender':['Male','Female','Male','Male']
     }
data_df=pd.DataFrame(data,index=['one','two','three','four'])
data_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Year</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>Chulmin</td>
      <td>2011</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>two</th>
      <td>Eunkyung</td>
      <td>2016</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>three</th>
      <td>Jinwoong</td>
      <td>2015</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>four</th>
      <td>Soobeom</td>
      <td>2015</td>
      <td>Male</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_df.iloc[0,0]
```




    'Chulmin'




```python
data_df.iloc[0,'Name'] #명칭 입력시 오류
data_df.iloc['one,0']
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexing.py in _has_valid_tuple(self, key)
        753             try:
    --> 754                 self._validate_key(k, i)
        755             except ValueError as err:


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexing.py in _validate_key(self, key, axis)
       1425         else:
    -> 1426             raise ValueError(f"Can only index by location with a [{self._valid_types}]")
       1427 


    ValueError: Can only index by location with a [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array]

    
    The above exception was the direct cause of the following exception:


    ValueError                                Traceback (most recent call last)

    /var/folders/hg/s9q7g8q94b13_96p488f3z3m0000gn/T/ipykernel_81587/2080460278.py in <module>
    ----> 1 data_df.iloc[0,'Name'] #명칭 입력시 오류
          2 data_df.iloc['one,0']


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexing.py in __getitem__(self, key)
        923                 with suppress(KeyError, IndexError):
        924                     return self.obj._get_value(*key, takeable=self._takeable)
    --> 925             return self._getitem_tuple(key)
        926         else:
        927             # we by definition only have the 0th axis


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
       1504     def _getitem_tuple(self, tup: tuple):
       1505 
    -> 1506         self._has_valid_tuple(tup)
       1507         with suppress(IndexingError):
       1508             return self._getitem_lowerdim(tup)


    /opt/anaconda3/envs/boost/lib/python3.8/site-packages/pandas/core/indexing.py in _has_valid_tuple(self, key)
        754                 self._validate_key(k, i)
        755             except ValueError as err:
    --> 756                 raise ValueError(
        757                     "Location based indexing can only have "
        758                     f"[{self._valid_types}] types"


    ValueError: Location based indexing can only have [integer, integer slice (START point is INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types



```python
data_df_reset=data_df.reset_index()
data_df_reset=data_df_reset.rename(columns={'index':'old_index'})
data_df_reset.index = data_df_reset.index+1
print(data_df_reset)
data_df_reset.iloc[0,1] #index 값과 헷갈리지 않음
```

      old_index      Name  Year  Gender
    1       one   Chulmin  2011    Male
    2       two  Eunkyung  2016  Female
    3     three  Jinwoong  2015    Male
    4      four   Soobeom  2015    Male





    'Chulmin'



#### loc[] - 명칭 기반 Indexing


```python
data_df.loc['one','Name']
```




    'Chulmin'




```python
data_df_reset.loc[1,'Name'] #이 경우에는 인덱스 값으로 1이 존재하므로 가능
```




    'Chulmin'




```python
print('위치 기반 iloc slicing\n', data_df.iloc[0:1,0],'\n') # 0부터 1-1까지
print('명칭 기반 loc slicing\n', data_df.loc['one':'two','Name']) #'one'부터 'two'까지 ('two'-1이 불가능하므로)
```

    위치 기반 iloc slicing
     one    Chulmin
    Name: Name, dtype: object 
    
    명칭 기반 loc slicing
     one     Chulmin
    two    Eunkyung
    Name: Name, dtype: object



```python
print(data_df_reset.loc[1:2,'Name'])
```

    1     Chulmin
    2    Eunkyung
    Name: Name, dtype: object


#### Boolean Indexing


```python

```
