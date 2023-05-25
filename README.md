# Movie Recommendation System
### Eva Burns

## Problem Statement

Everyone has encountered this problem before: you sit down to watch a movie, have no idea what to watch or what you think you would like. To solve this, I will create a recommendation system using about 6 million ratings from 270,000 users. The recommendations for each user are created by predicting what the user will rate each movie, and recommending the highest predicted rated movies.

The dataset was found at [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) on Kaggle. This dataset consists of more files on the content information about the movies such as cast and crew, but I will focus on the following files for this recommender:

[movies_metadata.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): The main Movies Metadata file. Contains information on 45,000 movies found on TMDB. Relevant features include id, title, genres, popularity, vote_average, and vote_count

[ratings_small.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv): The subset of 100,000 ratings from 700 users on 9,000 movies. This is used for development and testing the model before using more data.

[ratings.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings.csv): The full 26 million ratings from 270,000 users for all 45,000 movies.

The ratings.csv file is too large for GitHub, so the data will have to be accessed from Kaggle.

## Assumptions/Hypotheses about data and model

Because ratings are user inputted, there are biases involved in the data. There are certain groups of people who may leave ratings on movies: critics and people who feel very strongly about the movie (both positively and negatively). For the purposes of this project, I will assume that the ratings are representative of the general population.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate
from ast import literal_eval
from zipfile import ZipFile
import os
import random
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
```


```python
SOURCE_PATH = './data'

if (not os.path.exists(f'{SOURCE_PATH}/movies_metadata.csv')) | (not os.path.exists(f'{SOURCE_PATH}/ratings_small.csv')) | (not os.path.exists(f'{SOURCE_PATH}/ratings.csv')):
    zip = ZipFile(f'{SOURCE_PATH}/movie_data.zip')
    zip.extractall(path = SOURCE_PATH)
    zip.close()

    os.rmdir(f'{SOURCE_PATH}/__MACOSX')
    
movies_metadata = pd.read_csv(f'{SOURCE_PATH}/movies_metadata.csv', low_memory=False)
ratings_small = pd.read_csv(f'{SOURCE_PATH}/ratings_small.csv') # Used for testing and building purposes
ratings = pd.read_csv(f'{SOURCE_PATH}/ratings.csv')
```

### Data Cleaning

From the movies metadata dataset, I will drop some of the columns that will not be useful for this project.


```python
movies_md = movies_metadata.copy()
movies_md = movies_md.drop(["homepage", "status", "video", "poster_path", "belongs_to_collection", 'adult', 'original_language', 'production_countries', 'original_title'], axis=1)
```

I also would like to convert the id column to int and the popularity column to float. However, there were some issues with the data imputation, and have some weird strings as a data point. Those will be made into null values and then dropped.


```python
def convert_float(x):
    try:
        return float(x)
    except:
        return np.nan

movies_md['popularity'] = movies_md['popularity'].apply(convert_float)
movies_md['id'] = movies_md['id'].apply(convert_float)

n_orig = len(movies_md['popularity'])

movies_md = movies_md[(movies_md['popularity'].notna()) & (movies_md['id'].notna())]
movies_md['id'] = movies_md['id'].astype(int)
n_new = len(movies_md['popularity'])

print(f"Number rows dropped: {n_orig - n_new}")
```

    Number rows dropped: 6


I will also be evaluating the genres of the movies later, so I will convert them from a dictionary/json format to a list.


```python
movies_md['genres'] = movies_md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
```

## Exploratory Data Analysis

I will begin by examining the most popular movies. A popularity score is calculated by TMDB basically by how much users interact with the movie (rating, viewing, favoriting, etc.)


```python
simplified_md = movies_md[['id','title', 'popularity', 'genres']].sort_values(by='popularity', ascending=False)

grouped = ratings.groupby('movieId')

mean_ratings = grouped.mean()['rating']
num_ratings = grouped.count()['rating']

movie_ratings = pd.DataFrame({'avg_rating':mean_ratings, 'ratings_count': num_ratings}).reset_index()

simplified_md = simplified_md.merge(movie_ratings, left_on='id', right_on='movieId').drop('movieId', axis=1)

simplified_md.head(15)
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
      <th>id</th>
      <th>title</th>
      <th>popularity</th>
      <th>genres</th>
      <th>avg_rating</th>
      <th>ratings_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>680</td>
      <td>Pulp Fiction</td>
      <td>140.950236</td>
      <td>[Thriller, Crime]</td>
      <td>3.552970</td>
      <td>1246</td>
    </tr>
    <tr>
      <th>1</th>
      <td>155</td>
      <td>The Dark Knight</td>
      <td>123.167259</td>
      <td>[Drama, Action, Crime, Thriller]</td>
      <td>3.395375</td>
      <td>1319</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>Blade Runner</td>
      <td>96.272374</td>
      <td>[Science Fiction, Drama, Thriller]</td>
      <td>3.173709</td>
      <td>1278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>119450</td>
      <td>Dawn of the Planet of the Apes</td>
      <td>75.385211</td>
      <td>[Science Fiction, Action, Drama, Thriller]</td>
      <td>3.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>550</td>
      <td>Fight Club</td>
      <td>63.869599</td>
      <td>[Drama]</td>
      <td>3.083261</td>
      <td>3477</td>
    </tr>
    <tr>
      <th>5</th>
      <td>118340</td>
      <td>Guardians of the Galaxy</td>
      <td>53.291601</td>
      <td>[Action, Science Fiction, Adventure]</td>
      <td>4.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>278</td>
      <td>The Shawshank Redemption</td>
      <td>51.645403</td>
      <td>[Drama, Crime]</td>
      <td>3.021647</td>
      <td>1178</td>
    </tr>
    <tr>
      <th>7</th>
      <td>13</td>
      <td>Forrest Gump</td>
      <td>48.307194</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>3.326442</td>
      <td>1838</td>
    </tr>
    <tr>
      <th>8</th>
      <td>22</td>
      <td>Pirates of the Caribbean: The Curse of the Bla...</td>
      <td>47.326665</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>3.300608</td>
      <td>11026</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>Star Wars</td>
      <td>42.149697</td>
      <td>[Adventure, Action, Science Fiction]</td>
      <td>3.660591</td>
      <td>19475</td>
    </tr>
    <tr>
      <th>10</th>
      <td>424</td>
      <td>Schindler's List</td>
      <td>41.725123</td>
      <td>[Drama, History, War]</td>
      <td>2.707179</td>
      <td>794</td>
    </tr>
    <tr>
      <th>11</th>
      <td>238</td>
      <td>The Godfather</td>
      <td>41.109264</td>
      <td>[Drama, Crime]</td>
      <td>3.164196</td>
      <td>877</td>
    </tr>
    <tr>
      <th>12</th>
      <td>129</td>
      <td>Spirited Away</td>
      <td>41.048867</td>
      <td>[Fantasy, Adventure, Animation, Family]</td>
      <td>3.170000</td>
      <td>200</td>
    </tr>
    <tr>
      <th>13</th>
      <td>637</td>
      <td>Life Is Beautiful</td>
      <td>39.394970</td>
      <td>[Comedy, Drama]</td>
      <td>2.846865</td>
      <td>6922</td>
    </tr>
    <tr>
      <th>14</th>
      <td>671</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>38.187238</td>
      <td>[Adventure, Fantasy, Family]</td>
      <td>3.658049</td>
      <td>7330</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_pop = simplified_md.head(10)

plt.barh(top_pop['title'],top_pop['popularity'])
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
```




    Text(0.5, 1.0, 'Popular Movies')




    
![png](output_11_1.png)
    


Below is the distribution of genres for the top 25 most popular movies.


```python
def count_genres(genre_series):
    total_ct = 0
    counts = dict()
    counts['Other'] = 0
    for genres in genre_series:
        for genre in genres:
            if genre not in counts.keys():
                counts[genre] = 1
            else: 
                counts[genre] += 1
            total_ct += 1
    
    for key in counts.keys():
        if counts[key] < .02 * total_ct:
            counts['Other'] += 1
            counts[key] = 0
    
    filtered_dict = {k:v for k,v in counts.items() if v != 0}
    return dict(sorted(filtered_dict.items(), key=lambda x:x[1], reverse=True))

popular_genres = count_genres(simplified_md.head(25)['genres'])

plt.pie(popular_genres.values(), labels=popular_genres.keys(), autopct='%1.1f%%')
plt.title("Genres of Most Popular Movies")
plt.axis('equal')
plt.show()
```


    
![png](output_13_0.png)
    


Now, I will examine the highest rated movies. I will do a few transformations to get a better idea of what is considered the highest rated movie. There are movies that only have one rating, so giving them the same weight as another movie with 10,000 ratings seems unreasonable. So first, I will filter out the movies in the lowest 10% of number of ratings. Then, I will calculate a weighted rating inspired by this formula used by IMDB:

$$\text{Weighted Rating (WR)} = \left( \frac{v}{v+m} \cdot R\right) + \left( \frac{m}{v+m} \cdot C\right)$$

$v$ is the number of votes for the movie

$m$ is the minimum votes required to be listed in the chart

$R$ is the average rating of the movie

$C$ is the mean vote across the whole report


```python
C = simplified_md['avg_rating'].mean()
m = simplified_md['ratings_count'].quantile(0.9)

def weighted_rating(x, m=m, C=C):
    v = x['ratings_count']
    R = x['avg_rating']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

filtered_movies = simplified_md.copy().loc[simplified_md['ratings_count'] >= m]

filtered_movies['score'] = filtered_movies.apply(weighted_rating, axis=1)

filtered_movies = filtered_movies.sort_values(by="score", ascending=False)
filtered_movies.head(15)
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
      <th>id</th>
      <th>title</th>
      <th>popularity</th>
      <th>genres</th>
      <th>avg_rating</th>
      <th>ratings_count</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2271</th>
      <td>318</td>
      <td>The Million Dollar Hotel</td>
      <td>4.938231</td>
      <td>[Drama, Thriller]</td>
      <td>4.429015</td>
      <td>91082</td>
      <td>4.382976</td>
    </tr>
    <tr>
      <th>925</th>
      <td>858</td>
      <td>Sleepless in Seattle</td>
      <td>10.234919</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>4.339811</td>
      <td>57070</td>
      <td>4.272782</td>
    </tr>
    <tr>
      <th>2526</th>
      <td>527</td>
      <td>Once Were Warriors</td>
      <td>4.025276</td>
      <td>[Drama]</td>
      <td>4.266531</td>
      <td>67662</td>
      <td>4.212948</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>2959</td>
      <td>License to Wed</td>
      <td>7.102076</td>
      <td>[Comedy]</td>
      <td>4.230716</td>
      <td>60024</td>
      <td>4.172561</td>
    </tr>
    <tr>
      <th>72</th>
      <td>296</td>
      <td>Terminator 3: Rise of the Machines</td>
      <td>20.818907</td>
      <td>[Action, Thriller, Science Fiction]</td>
      <td>4.169975</td>
      <td>87901</td>
      <td>4.131812</td>
    </tr>
    <tr>
      <th>768</th>
      <td>593</td>
      <td>Solaris</td>
      <td>11.059785</td>
      <td>[Drama, Science Fiction, Adventure, Mystery]</td>
      <td>4.152246</td>
      <td>84078</td>
      <td>4.113090</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>912</td>
      <td>The Thomas Crown Affair</td>
      <td>9.444818</td>
      <td>[Romance, Crime, Thriller, Drama]</td>
      <td>4.214393</td>
      <td>30043</td>
      <td>4.105640</td>
    </tr>
    <tr>
      <th>4844</th>
      <td>58559</td>
      <td>Confession of a Child of the Century</td>
      <td>0.778480</td>
      <td>[Drama]</td>
      <td>4.182071</td>
      <td>39600</td>
      <td>4.100037</td>
    </tr>
    <tr>
      <th>1638</th>
      <td>750</td>
      <td>Murder She Said</td>
      <td>7.261845</td>
      <td>[Drama, Crime, Mystery, Comedy]</td>
      <td>4.213030</td>
      <td>28280</td>
      <td>4.098358</td>
    </tr>
    <tr>
      <th>2028</th>
      <td>260</td>
      <td>The 39 Steps</td>
      <td>5.865697</td>
      <td>[Action, Thriller, Mystery]</td>
      <td>4.132299</td>
      <td>77045</td>
      <td>4.090546</td>
    </tr>
    <tr>
      <th>640</th>
      <td>1213</td>
      <td>The Talented Mr. Ripley</td>
      <td>11.655020</td>
      <td>[Thriller, Crime, Drama]</td>
      <td>4.178289</td>
      <td>33987</td>
      <td>4.084257</td>
    </tr>
    <tr>
      <th>2544</th>
      <td>4226</td>
      <td>Shriek If You Know What I Did Last Friday the ...</td>
      <td>3.956594</td>
      <td>[Comedy]</td>
      <td>4.157078</td>
      <td>40706</td>
      <td>4.078999</td>
    </tr>
    <tr>
      <th>5244</th>
      <td>5618</td>
      <td>Cousin, Cousine</td>
      <td>0.624726</td>
      <td>[Romance, Comedy]</td>
      <td>4.202589</td>
      <td>20855</td>
      <td>4.054166</td>
    </tr>
    <tr>
      <th>3940</th>
      <td>4993</td>
      <td>5 Card Stud</td>
      <td>1.372254</td>
      <td>[Action, Western, Thriller]</td>
      <td>4.104167</td>
      <td>56827</td>
      <td>4.049945</td>
    </tr>
    <tr>
      <th>162</th>
      <td>608</td>
      <td>Men in Black II</td>
      <td>16.775716</td>
      <td>[Action, Adventure, Comedy, Science Fiction]</td>
      <td>4.105347</td>
      <td>52474</td>
      <td>4.046826</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_score = filtered_movies.head(10)

plt.barh(top_score['title'],top_score['score'])
plt.gca().invert_yaxis()
plt.xlabel("Weighted Score")
plt.title("High Scoring Movies")
```




    Text(0.5, 1.0, 'High Scoring Movies')




    
![png](output_16_1.png)
    



```python
high_score_genres = count_genres(filtered_movies.head(25)['genres'])

plt.pie(high_score_genres.values(), labels=high_score_genres.keys(), autopct='%1.1f%%')
plt.title("Genres of Highly Scored Movies")
plt.axis('equal')
plt.show()
```


    
![png](output_17_0.png)
    


## Feature Engineering & Transformations

There are no transformations needed for the model I will use for this project. All I have to do is convert the ratings data into a readable format for the model I will use which I will explain in the next section. The `Reader` and `Dataset.load_from_df` comes from the `surprise` package where the data is converted to a list of tuples.


```python
reader = Reader()

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.raw_ratings[0:10]
```




    [(1, 110, 1.0, None),
     (1, 147, 4.5, None),
     (1, 858, 5.0, None),
     (1, 1221, 5.0, None),
     (1, 1246, 5.0, None),
     (1, 1968, 4.0, None),
     (1, 2762, 4.5, None),
     (1, 2918, 5.0, None),
     (1, 2959, 4.0, None),
     (1, 4226, 4.0, None)]



## Proposed Approaches (Model)

### Single Value Decomposition

In a previous homework assignment, we used Collabrative Filtering (CF) to build a recommendations system. However, CF has a couple issues. The primary challenge at hand is scalability, where the computational demands increase as the number of users and movies grows. Furthermore, sparsity poses another issue. In certain scenarios, there could be millions of users, and the similarity between two seemingly dissimilar movies might be remarkably high due to the presence of a single user who ranked them both similarly.

One approach to address the scalability and sparsity challenges posed by CF is to utilize a latent factor model to capture the similarity between users and items. The aim is to transform the recommendation problem into an optimization problem, specifically focusing on accurately predicting movie ratings for a given user.

Latent factors encompasses the inherent properties or concepts associated with users or movies. For this project, the latent factor represents the rating the users gave the movies. Through Singular Value Decomposition (SVD), we reduce the dimensionality of the utility matrix by extracting its latent factors. This process involves mapping each user and movie onto a latent space with a dimension of 'r'. Consequently, it facilitates a more meaningful understanding of the relationship between users and movies, making them directly comparable.

This SVD model was built using the [Surprise package](https://surpriselib.com/).

Hug, N., (2020). Surprise: A Python library for recommender systems. Journal of Open Source Software, 5(52), 2174, https://doi.org/10.21105/joss.02174


```python
svd = SVD()
```

## Proposed Solution (Model Selection)

To train the model, I will employ a cross validation of 5 folds, using RMSE and MAE as accuracy metrics.


```python
import time
start = time.time()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

end = time.time()
print(f"\nTime elasped: {round(end-start, 3)} seconds ({round(len(ratings)/(end-start), 3)} ratings per second)")

trainset = data.build_full_trainset()
testset = trainset.build_testset()

predictions = svd.test(testset)

rmse = accuracy.rmse(predictions, verbose=False) 
mse = accuracy.mse(predictions, verbose=False) 
mae = accuracy.mae(predictions, verbose=False) 

svd.fit(trainset)
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.7958  0.7959  0.7964  0.7967  0.7962  0.7962  0.0004  
    MAE (testset)     0.6018  0.6021  0.6023  0.6026  0.6022  0.6022  0.0003  
    Fit time          185.94  385.93  365.98  394.30  344.17  335.26  76.65   
    Test time         152.14  121.58  140.40  127.49  118.78  132.08  12.49   
    
    Time elasped: 3240.326 seconds (8031.38 ratings per second)





    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x14addb340>



Now that I have a model that will predict what a user will rate a specific movie, I will now create a list of recommended movies for a given user based on what the SVD predict will be their highest rated movies. 

I do this by going through every movie in the database, and getting the predicted rating for that user/movie pairing. Once I have done that for every movie, I will sort the ratings so that the highest predicted rated movie is recommended to the user. Note that all movies the user has already rated have been excluded from the list. I will demonstrate the recommender in the next section.


```python
def get_recommendations_for_user(user_id):
    est_ratings = []
    for i, row in filtered_movies.iterrows():
        movie_id = int(row['id'])
        title = row['title']
        est_rat = svd.predict(user_id, movie_id).est
        est_ratings += [[movie_id, row['title'], est_rat, row['genres']]]

    recommendation = pd.DataFrame(est_ratings, columns=["movieId", "title", "estimated_rating", 'genres'])
    recommendation = recommendation[~recommendation['movieId'].isin(ratings[ratings['userId'] == user_id]['movieId'])]
    
    return recommendation.sort_values(by='estimated_rating', ascending=False)
```

## Results (Accuracy) and Learnings from the methodology

I will evaluate the accuracy of the model on the test set using RMSE, MSE, MAE. The model predicts what the user will rate a movie, so I can calculate how accurate the SVD model is on movies that have already been rated. 


```python
print("RMSE:", round(rmse, 4))
print("MSE:", round(mse, 4))
print("MAE:", round(mae, 4))
```

    RMSE: 0.7031
    MSE: 0.4943
    MAE: 0.5338


The accuracy metrics are quite low for this type of model, so I believe it is sucessfully predicting user ratings. There is always room for improvement, though.

### Example user recommendation

I've chosen User 1 to demonstrate the recommender. Below is what they have rated the movies they have watched, sorted by rating.


```python
user_id = 1

user_hist = ratings[ratings['userId'] == user_id].merge(movies_md[['id', 'title', 'genres']], left_on='movieId', right_on="id")
user_hist = user_hist.sort_values(by='rating', ascending=False)
user_hist
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>id</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>858</td>
      <td>5.0</td>
      <td>1425941523</td>
      <td>858</td>
      <td>Sleepless in Seattle</td>
      <td>[Comedy, Drama, Romance]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1246</td>
      <td>5.0</td>
      <td>1425941556</td>
      <td>1246</td>
      <td>Rocky Balboa</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>96821</td>
      <td>5.0</td>
      <td>1425941382</td>
      <td>96821</td>
      <td>Caesar Must Die</td>
      <td>[Drama, Documentary]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>147</td>
      <td>4.5</td>
      <td>1425942435</td>
      <td>147</td>
      <td>The 400 Blows</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2762</td>
      <td>4.5</td>
      <td>1425941300</td>
      <td>2762</td>
      <td>Young and Innocent</td>
      <td>[Drama, Crime]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1968</td>
      <td>4.0</td>
      <td>1425942148</td>
      <td>1968</td>
      <td>Fools Rush In</td>
      <td>[Drama, Comedy, Romance]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>2959</td>
      <td>4.0</td>
      <td>1425941601</td>
      <td>2959</td>
      <td>License to Wed</td>
      <td>[Comedy]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>4226</td>
      <td>4.0</td>
      <td>1425942228</td>
      <td>4226</td>
      <td>Shriek If You Know What I Did Last Friday the ...</td>
      <td>[Comedy]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>58559</td>
      <td>4.0</td>
      <td>1425942007</td>
      <td>58559</td>
      <td>Confession of a Child of the Century</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>54503</td>
      <td>3.5</td>
      <td>1425941313</td>
      <td>54503</td>
      <td>The Mystery of Chess Boxing</td>
      <td>[Action, Foreign]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>110</td>
      <td>1.0</td>
      <td>1425941529</td>
      <td>110</td>
      <td>Three Colors: Red</td>
      <td>[Drama, Mystery, Romance]</td>
    </tr>
  </tbody>
</table>
</div>



Now, I will use my recommender to recommend 10 movies to the user using the method described above.


```python
user_rec = get_recommendations_for_user(user_id).head(10)
user_rec
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
      <th>movieId</th>
      <th>title</th>
      <th>estimated_rating</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>71</th>
      <td>306</td>
      <td>Beverly Hills Cop III</td>
      <td>5.000000</td>
      <td>[Action, Comedy, Crime]</td>
    </tr>
    <tr>
      <th>36</th>
      <td>44555</td>
      <td>A Woman, a Gun and a Noodle Shop</td>
      <td>4.904216</td>
      <td>[Comedy, Drama, Thriller]</td>
    </tr>
    <tr>
      <th>101</th>
      <td>307</td>
      <td>Rome, Open City</td>
      <td>4.895459</td>
      <td>[Drama, History]</td>
    </tr>
    <tr>
      <th>212</th>
      <td>28</td>
      <td>Apocalypse Now</td>
      <td>4.848280</td>
      <td>[Drama, War]</td>
    </tr>
    <tr>
      <th>53</th>
      <td>17</td>
      <td>The Dark</td>
      <td>4.840669</td>
      <td>[Horror, Thriller, Mystery]</td>
    </tr>
    <tr>
      <th>67</th>
      <td>246</td>
      <td>Zatoichi</td>
      <td>4.793591</td>
      <td>[Adventure, Drama, Action]</td>
    </tr>
    <tr>
      <th>161</th>
      <td>2064</td>
      <td>While You Were Sleeping</td>
      <td>4.784361</td>
      <td>[Comedy, Drama, Romance]</td>
    </tr>
    <tr>
      <th>47</th>
      <td>5971</td>
      <td>We're No Angels</td>
      <td>4.777725</td>
      <td>[Comedy, Crime, Drama]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019</td>
      <td>Hard Target</td>
      <td>4.757174</td>
      <td>[Action, Adventure, Crime, Thriller]</td>
    </tr>
    <tr>
      <th>146</th>
      <td>3089</td>
      <td>Red River</td>
      <td>4.756867</td>
      <td>[Western]</td>
    </tr>
  </tbody>
</table>
</div>



The highest estimated rating was 5 for Bevery Hills Cop III for User 1. Comparing the genre breakdown of User 1's actual ratings versus the recommended movies makes sense. The user liked dramas and comedies the most, which is reflected in the recommended movies.


```python
if len(user_hist) < 10:
    user_genre_count = count_genres(user_hist['genres'])
else:
    user_genre_count = count_genres(user_hist.head(10)['genres'])

plt.subplot(1, 2, 1)
plt.pie(user_genre_count.values(), labels=user_genre_count.keys(), autopct='%1.1f%%')
plt.title(f"Genres of the Top 10 Highest Rated Movies for User {user_hist.iloc[0,0]}")
plt.axis('equal')
plt.show()

rec_genre_count = count_genres(user_rec['genres'])

plt.subplot(1, 2, 2)
plt.pie(rec_genre_count.values(), labels=rec_genre_count.keys(), autopct='%1.1f%%')
plt.title(f"Genres of the Top 10 Recommended Movies for User {user_hist.iloc[0,0]}")
plt.axis('equal')
plt.show()
```


    
![png](output_33_0.png)
    



    
![png](output_33_1.png)
    


## Future Work

There was a good portion of the data from kaggle I did not use, mainly information about the movies such as cast, crew, keywords from the description, and genre. Next steps would be to make a hybrid recommender that incorporates the svd model already made with some of the movie information. For example, based on the movies the user gave a high rating for, find movies that other users also liked (SVD). Then, find movies that have similar plots, cast, or genre to the movies the user already liked.

I think incorporating these multiple aspects could make an even more accurate recommender.

Another note is that the highest recommended movie for User 1 was Beverly Hills Cop III, which is a sequel. User 1 has not seen the original movies, so it does not make sense to recommend that movie. An adjustment to the recommender should be made to filter out sequels if the user has not seen the movies before it.


```python
os.remove(f'{SOURCE_PATH}/movies_metadata.csv')
os.remove(f'{SOURCE_PATH}/ratings_small.csv')
os.remove(f'{SOURCE_PATH}/ratings.csv')
```
