```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from ast import literal_eval
from zipfile import ZipFile
import os
import random
```

# Movie Recommendation System
### Eva Burns

## Problem Statement

Everyone has encountered this problem before: you sit down to watch a movie, have no idea what to watch or what you think you would like. To solve this, I will create a recommendation system using about 9.6 million ratings from 100,000 users. The recommendations for each user are created by predicting what the user will rate each movie, and recommending the highest predicted rated movies.

The dataset was found at [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) on Kaggle. This dataset consists of more files on the content information about the movies such as cast and crew, but I will focus on the following files for this recommender:

[movies_metadata.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv): The main Movies Metadata file. Contains information on 45,000 movies found on TMDB. Relevant features include id, title, genres, popularity, vote_average, and vote_count

[ratings_small.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv): The subset of 100,000 ratings from 700 users on 9,000 movies. This is used for development and testing the model before using more data.

[ratings.csv](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings.csv): The full 26 million ratings from 270,000 users for all 45,000 movies.

## Assumptions/Hypotheses about data and model

Because ratings are user inputted, there are biases involved in the data. There are certain groups of people who may leave ratings on movies: critics and people who feel very strongly about the movie (both positively and negatively). For the purposes of this project, I will assume that the ratings are representative of the general population.


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

I will be taking a subset of the full ratings dataset. This will be done by taking a random sample without replacement of size 100,000 from all of the unique user ids and getting their ratings. The reason I did not just randomly split the entire ratings dataset is because I want every user represented in the dataset to have their full ratings. That is more useful to the model to know 


```python
random.seed(123)
ids = random.sample(list(set(ratings['userId'])), 10000)
ratings_medium = ratings[ratings['userId'].isin(ids)] 
len(ratings_medium)
```




    954874



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




    
![png](output_13_1.png)
    


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


    
![png](output_15_0.png)
    


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




    
![png](output_18_1.png)
    



```python
high_score_genres = count_genres(filtered_movies.head(25)['genres'])

plt.pie(high_score_genres.values(), labels=high_score_genres.keys(), autopct='%1.1f%%')
plt.title("Genres of Highly Scored Movies")
plt.axis('equal')
plt.show()
```


    
![png](output_19_0.png)
    


## Feature Engineering & Transformations

## Proposed Approaches (Model) with checks for overfitting/underfitting

Collaborative Filtering


```python
reader = Reader()

data = Dataset.load_from_df(ratings_medium[['userId', 'movieId', 'rating']], reader)

svd = SVD()
```

## Proposed Solution (Model Selection) with regularization, if needed


```python
import time
start = time.time()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

end = time.time()
print(f"\nTime elasped: {round(end-start, 3)} seconds ({round(len(ratings_medium)/(end-start), 3)} ratings per second)", )
```

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
    
                      Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
    RMSE (testset)    0.8608  0.8581  0.8581  0.8567  0.8565  0.8580  0.0015  
    MAE (testset)     0.6558  0.6540  0.6542  0.6527  0.6527  0.6539  0.0012  
    Fit time          4.19    4.77    4.28    4.23    4.55    4.41    0.22    
    Test time         0.65    0.69    0.70    0.66    0.68    0.68    0.02    
    
    Time elasped: 29.612 seconds (32245.69 ratings per second)


## Results (Accuracy) and Learnings from the methodology


```python
trainset = data.build_full_trainset()

svd.fit(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x1088535e0>




```python
user_id = ids[100]

user_hist = ratings_medium[ratings_medium['userId'] == user_id].merge(movies_md[['id', 'title', 'genres']], left_on='movieId', right_on="id")
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
      <th>0</th>
      <td>227122</td>
      <td>345</td>
      <td>5.0</td>
      <td>1246787949</td>
      <td>345</td>
      <td>Eyes Wide Shut</td>
      <td>[Mystery, Drama]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>227122</td>
      <td>924</td>
      <td>5.0</td>
      <td>1246787779</td>
      <td>924</td>
      <td>Dawn of the Dead</td>
      <td>[Fantasy, Horror, Action]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>227122</td>
      <td>44555</td>
      <td>5.0</td>
      <td>1246788227</td>
      <td>44555</td>
      <td>A Woman, a Gun and a Noodle Shop</td>
      <td>[Comedy, Drama, Thriller]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>227122</td>
      <td>8973</td>
      <td>5.0</td>
      <td>1246789196</td>
      <td>8973</td>
      <td>Lord of Illusions</td>
      <td>[Mystery, Horror, Thriller]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>227122</td>
      <td>3083</td>
      <td>5.0</td>
      <td>1246788054</td>
      <td>3083</td>
      <td>Mr. Smith Goes to Washington</td>
      <td>[Comedy, Drama]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>227122</td>
      <td>39183</td>
      <td>4.5</td>
      <td>1246788609</td>
      <td>39183</td>
      <td>Once in a Lifetime: The Extraordinary Story of...</td>
      <td>[Documentary]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>227122</td>
      <td>43376</td>
      <td>4.5</td>
      <td>1246788264</td>
      <td>43376</td>
      <td>Diary of a Country Priest</td>
      <td>[Drama]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>227122</td>
      <td>1103</td>
      <td>4.0</td>
      <td>1246787622</td>
      <td>1103</td>
      <td>Escape from New York</td>
      <td>[Science Fiction, Action]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>227122</td>
      <td>4967</td>
      <td>4.0</td>
      <td>1246788380</td>
      <td>4967</td>
      <td>Keeping the Faith</td>
      <td>[Comedy]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>227122</td>
      <td>922</td>
      <td>4.0</td>
      <td>1246787620</td>
      <td>922</td>
      <td>Dead Man</td>
      <td>[Drama, Fantasy, Western]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>227122</td>
      <td>44694</td>
      <td>4.0</td>
      <td>1246788599</td>
      <td>44694</td>
      <td>Big Time</td>
      <td>[Documentary, Drama, Music, Romance]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>227122</td>
      <td>2186</td>
      <td>3.5</td>
      <td>1246788239</td>
      <td>2186</td>
      <td>Within the Woods</td>
      <td>[Horror]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>227122</td>
      <td>2300</td>
      <td>3.5</td>
      <td>1246787600</td>
      <td>2300</td>
      <td>Space Jam</td>
      <td>[Animation, Comedy, Drama, Family, Fantasy]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>227122</td>
      <td>508</td>
      <td>3.5</td>
      <td>1246788638</td>
      <td>508</td>
      <td>Love Actually</td>
      <td>[Comedy, Romance, Drama]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>227122</td>
      <td>54796</td>
      <td>3.5</td>
      <td>1246787769</td>
      <td>54796</td>
      <td>Thicker than Water</td>
      <td>[TV Movie, Drama]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>227122</td>
      <td>63876</td>
      <td>3.5</td>
      <td>1246788625</td>
      <td>63876</td>
      <td>Circle of Love</td>
      <td>[Drama, Comedy, History, Romance]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>227122</td>
      <td>1729</td>
      <td>2.5</td>
      <td>1246788621</td>
      <td>1729</td>
      <td>The Forbidden Kingdom</td>
      <td>[Action, Adventure, Fantasy]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>227122</td>
      <td>923</td>
      <td>2.0</td>
      <td>1246788268</td>
      <td>923</td>
      <td>Dawn of the Dead</td>
      <td>[Horror]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>227122</td>
      <td>750</td>
      <td>2.0</td>
      <td>1290984227</td>
      <td>750</td>
      <td>Murder She Said</td>
      <td>[Drama, Crime, Mystery, Comedy]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>227122</td>
      <td>58559</td>
      <td>2.0</td>
      <td>1246787578</td>
      <td>58559</td>
      <td>Confession of a Child of the Century</td>
      <td>[Drama]</td>
    </tr>
  </tbody>
</table>
</div>




```python
if len(user_hist) < 10:
    user_genre_count = count_genres(user_hist['genres'])
else:
    user_genre_count = count_genres(user_hist.head(10)['genres'])

plt.pie(user_genre_count.values(), labels=user_genre_count.keys(), autopct='%1.1f%%')
plt.axis('equal')
plt.show()
```


    
![png](output_28_0.png)
    



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
      <th>105</th>
      <td>905</td>
      <td>Pandora's Box</td>
      <td>4.393716</td>
      <td>[Drama, Thriller, Romance]</td>
    </tr>
    <tr>
      <th>46</th>
      <td>953</td>
      <td>Madagascar</td>
      <td>4.338538</td>
      <td>[Family, Animation]</td>
    </tr>
    <tr>
      <th>47</th>
      <td>5971</td>
      <td>We're No Angels</td>
      <td>4.243593</td>
      <td>[Comedy, Crime, Drama]</td>
    </tr>
    <tr>
      <th>30</th>
      <td>913</td>
      <td>The Thomas Crown Affair</td>
      <td>4.227379</td>
      <td>[Drama, Crime, Romance]</td>
    </tr>
    <tr>
      <th>215</th>
      <td>534</td>
      <td>Terminator Salvation</td>
      <td>4.169603</td>
      <td>[Action, Science Fiction, Thriller]</td>
    </tr>
    <tr>
      <th>109</th>
      <td>928</td>
      <td>Gremlins 2: The New Batch</td>
      <td>4.135304</td>
      <td>[Comedy, Horror, Fantasy]</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2762</td>
      <td>Young and Innocent</td>
      <td>4.079673</td>
      <td>[Drama, Crime]</td>
    </tr>
    <tr>
      <th>278</th>
      <td>3060</td>
      <td>The Big Parade</td>
      <td>4.070152</td>
      <td>[Drama, Romance, War]</td>
    </tr>
    <tr>
      <th>188</th>
      <td>290</td>
      <td>Barton Fink</td>
      <td>4.062039</td>
      <td>[Comedy, Drama]</td>
    </tr>
    <tr>
      <th>111</th>
      <td>909</td>
      <td>Meet Me in St. Louis</td>
      <td>4.051369</td>
      <td>[Comedy, Music, Romance]</td>
    </tr>
  </tbody>
</table>
</div>




```python
rec_genre_count = count_genres(user_rec['genres'])

plt.pie(rec_genre_count.values(), labels=rec_genre_count.keys(), autopct='%1.1f%%')
plt.axis('equal')
plt.show()
```


    
![png](output_30_0.png)
    


## Future Work

There was a good portion of the data from kaggle I did not use, mainly information about the movies such as cast, crew, keywords from the description, and genre. Next steps would be to make a hybrid recommender that incorporates the svd model already made with some of the movie information. For example, based on the movies the user gave a high rating for, find movies that other users also liked (SVD). Then, find movies that have similar plots, cast, or genre to the movies the user already liked.

I think incorporating these multiple aspects could make an even more accurate recommender.


```python
os.remove(f'{SOURCE_PATH}/movies_metadata.csv')
os.remove(f'{SOURCE_PATH}/ratings_small.csv')
os.remove(f'{SOURCE_PATH}/ratings.csv')
```
