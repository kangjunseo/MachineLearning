```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
```

## Numpy

##### -np.array(), shape
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



```python

```


```python

```
