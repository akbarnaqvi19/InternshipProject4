## Ali Akbar Naqvi
## Internship Project 4
## Movie Recommendation System
## Item Based Colaborative Filtering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

print(movies)
print(ratings.columns)

data = ratings.pivot(index="movieId", columns="userId", values="rating")
data.fillna(0, inplace=True)

print(data)

no_user_rated= ratings.groupby("movieId")['rating'].agg('count')
no_movies_rated=ratings.groupby("userId")['rating'].agg('count')

print(no_user_rated)
print(no_movies_rated)

plt.style.use("ggplot")
fig,axes= plt.subplots(1,1, figsize=(16,4))
plt.scatter(no_user_rated.index, no_user_rated, color="orange")
plt.axhline(y=10, color='green')
plt.xlabel("Movie")
plt.ylabel("No of Users Voted")
plt.show()

data=data.loc[no_user_rated[no_user_rated > 10].index, :]
print(data)

plt.style.use("ggplot")
fig,axes= plt.subplots(1,1, figsize=(16,4))
plt.scatter(no_movies_rated.index, no_movies_rated, color="orange")
plt.axhline(y=50, color='green')
plt.xlabel("Users")
plt.ylabel("No of Movies Voted")
plt.show()

data=data.loc[:,no_movies_rated[no_movies_rated > 50].index]
print(data)

csr_data= csr_matrix(data.values)
data.reset_index(inplace=True)
print(csr_data)