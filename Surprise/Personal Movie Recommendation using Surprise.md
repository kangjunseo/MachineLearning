```python
!pip install scikit-surprise
```

    Collecting scikit-surprise
      Downloading scikit-surprise-1.1.1.tar.gz (11.8 MB)
         |████████████████████████████████| 11.8 MB 5.8 MB/s            
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: joblib>=0.11 in /opt/anaconda3/lib/python3.8/site-packages (from scikit-surprise) (1.0.1)
    Requirement already satisfied: numpy>=1.11.2 in /opt/anaconda3/lib/python3.8/site-packages (from scikit-surprise) (1.20.1)
    Requirement already satisfied: scipy>=1.0.0 in /opt/anaconda3/lib/python3.8/site-packages (from scikit-surprise) (1.6.2)
    Requirement already satisfied: six>=1.10.0 in /opt/anaconda3/lib/python3.8/site-packages (from scikit-surprise) (1.15.0)
    Building wheels for collected packages: scikit-surprise
      Building wheel for scikit-surprise (setup.py) ... [?25ldone
    [?25h  Created wheel for scikit-surprise: filename=scikit_surprise-1.1.1-cp38-cp38-macosx_10_9_x86_64.whl size=765197 sha256=e98839ff1f5883a2adf0654cef7b1b805d5923040b74cc09ed73ed06cf02ef7c
      Stored in directory: /Users/kangjunseo/Library/Caches/pip/wheels/20/91/57/2965d4cff1b8ac7ed1b6fa25741882af3974b54a31759e10b6
    Successfully built scikit-surprise
    Installing collected packages: scikit-surprise
    Successfully installed scikit-surprise-1.1.1
    [33mWARNING: You are using pip version 21.3.1; however, version 22.0.3 is available.
    You should consider upgrading via the '/opt/anaconda3/bin/python -m pip install --upgrade pip' command.[0m


## Quick Tutorial

### Imports


```python
from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split
```

### Load Dataset and Model


```python
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25, random_state=0)
```


```python
algo = SVD()
algo.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x7ff3396821c0>



### Predict


```python
predictions = algo.test(testset)
print('prediction type :', type(predictions), ' size:',len(predictions))
print('prediction result : head 5')
predictions[:5]
```

    prediction type : <class 'list'>  size: 25000
    prediction result : head 5





    [Prediction(uid='120', iid='282', r_ui=4.0, est=3.5626046429554394, details={'was_impossible': False}),
     Prediction(uid='882', iid='291', r_ui=4.0, est=3.7138058873373647, details={'was_impossible': False}),
     Prediction(uid='535', iid='507', r_ui=5.0, est=4.020537064849165, details={'was_impossible': False}),
     Prediction(uid='697', iid='244', r_ui=5.0, est=3.6866699942368455, details={'was_impossible': False}),
     Prediction(uid='751', iid='385', r_ui=4.0, est=3.2370095167499215, details={'was_impossible': False})]




```python
# access to features in Prediction object
[(pred.uid, pred.iid, pred.est) for pred in predictions[:3]]
```




    [('120', '282', 3.5626046429554394),
     ('882', '291', 3.7138058873373647),
     ('535', '507', 4.020537064849165)]




```python
# user id, item id should be string
uid = str(196)
iid = str(302)
pred = algo.predict(uid,iid)
print(pred)
```

    user: 196        item: 302        r_ui = None   est = 4.08   {'was_impossible': False}



```python
accuracy.rmse(predictions)
```

    RMSE: 0.9465





    0.9465015971169856





## Apply to MovieLens Dataset

Dataset from https://grouplens.org/datasets/movielens/latest/

### Load OS File Data to Surprise Dataset


```python
import pandas as pd

ratings = pd.read_csv('./ml-latest-small/ratings.csv')
# Create new csv without index and header
ratings.to_csv('./ml-latest-small/ratings_noh.csv', index=False, header=False)
```


```python
from surprise import Reader

# Create Reader class
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5,5))
data = Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)
```

### Fit and Predict


```python
trainset, testset = train_test_split(data, test_size=.25, random_state=0)
algo = SVD(n_factors=50, random_state=0)
```


```python
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

    RMSE: 0.8682





    0.8681952927143516





## Advanced prediction

### Load Dataset at Pandas DataFrame


```python
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)
```

    RMSE: 0.8682





    0.8681952927143516



### Cross Validation and Parameter Tuning


```python
from surprise.model_selection import cross_validate

cross_validate(algo, data, measures=['RMSE','MAE'],cv=5,verbose=True)
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8725  0.8692  0.8735  0.8700  0.8710  0.8713  0.0016  
    MAE (testset)     0.6706  0.6688  0.6728  0.6691  0.6673  0.6697  0.0019  
    Fit time          3.69    3.70    3.71    3.68    3.69    3.69    0.01    
    Test time         0.07    0.08    0.07    0.07    0.07    0.07    0.00    





    {'test_rmse': array([0.8725097 , 0.8691865 , 0.8735371 , 0.87002875, 0.87101993]),
     'test_mae': array([0.67059773, 0.66875884, 0.67282806, 0.66908158, 0.66726814]),
     'fit_time': (3.687675952911377,
      3.6965129375457764,
      3.7148258686065674,
      3.67720890045166,
      3.6904780864715576),
     'test_time': (0.07318711280822754,
      0.07918882369995117,
      0.07237887382507324,
      0.07183003425598145,
      0.07297015190124512)}




```python
from surprise.model_selection import GridSearchCV

# make a dict of target parameters
param_grid = {'n_epochs':[20,40,60], 'n_factors':[50,100,200]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse','mae'],cv=3)
gs.fit(data)

# print best RMSE and best hyperparameter
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
```

    0.8762053629333026
    {'n_epochs': 20, 'n_factors': 50}




## Personalized Movie Recommendation

### Prepare Datasets


```python
# we'll use DatasetAutoFolds instead of train_test_split
# because we'll use every dataset just for training

from surprise.dataset import DatasetAutoFolds

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5,5))
data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv',reader = reader)

# Use full data for trainset
trainset = data_folds.build_full_trainset()
```


```python
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x7ff33f456940>




```python
movies = pd.read_csv('./ml-latest-small/movies.csv')

# target user : 9, target movie : 42
movieIds = ratings[ratings['userId']==9]['movieId']
if movieIds[movieIds==42].count()==0: print('User 9 does not have rating of movie 42')

print(movies[movies['movieId']==42])
```

    User 9 does not have rating of movie 42
        movieId                   title              genres
    38       42  Dead Presidents (1995)  Action|Crime|Drama


### Prediction for personal user


```python
uid, iid = str(9), str(42)
pred = algo.predict(uid,iid,verbose=True)
```

    user: 9          item: 42         r_ui = None   est = 3.13   {'was_impossible': False}



```python
def get_unseen_surprise(ratings, movies, userId):
    seen_movies = ratings[ratings['userId']==userId]['movieId'].tolist()
    total_movies = movies['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    
    print('seen movies :',len(seen_movies), 'unseen movies :',len(unseen_movies),'total movies :',len(total_movies))
    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9)
```

    seen movies : 46 unseen movies : 9696 total movies : 9742



```python
def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    predictions = [algo.predict(str(userId),str(movieId)) for movieId in unseen_movies]
    
    # predictions list features example
    # [Prediction(uid='9', iid='1', est=3.69), ...]
    
    # used for sorting by est value
    def sortkey_est(pred): return pred.est
    
    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]
    
    # Extract top_n movies' info
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_rating = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    
    top_movie_preds = [(id,title,rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating)]
    
    return top_movie_preds
```


```python
unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_preds = recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)

print('##### Top-10 Recommended Movie List #####')
for top_movie in top_movie_preds: print(top_movie[1],':',top_movie[2])
```

    seen movies : 46 unseen movies : 9696 total movies : 9742
    ##### Top-10 Recommended Movie List #####
    Usual Suspects, The (1995) : 4.306302135700814
    Star Wars: Episode IV - A New Hope (1977) : 4.281663842987387
    Pulp Fiction (1994) : 4.278152632122758
    Silence of the Lambs, The (1991) : 4.226073566460876
    Godfather, The (1972) : 4.1918097904381995
    Streetcar Named Desire, A (1951) : 4.154746591122658
    Star Wars: Episode V - The Empire Strikes Back (1980) : 4.122016128534504
    Star Wars: Episode VI - Return of the Jedi (1983) : 4.108009609093436
    Goodfellas (1990) : 4.083464936588478
    Glory (1989) : 4.07887165526957



```python

```
