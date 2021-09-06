```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
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

