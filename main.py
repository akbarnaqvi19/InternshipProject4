## Ali Akbar Naqvi
## Internship Project 4
## Movie Recommendation System
## Item Based Colaborative Filtering

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

data = ratings.pivot(index="movieId", columns="userId", values="rating")
data.fillna(0, inplace=True)

#print(data)

no_user_rated= ratings.groupby("movieId")['rating'].agg('count')
no_movies_rated=ratings.groupby("userId")['rating'].agg('count')

#print(no_user_rated)
#print(no_movies_rated)

data=data.loc[no_user_rated[no_user_rated > 10].index, :]
#print(data)

data=data.loc[:,no_movies_rated[no_movies_rated > 50].index]
#print(data)

csr_data= csr_matrix(data.values)
data.reset_index(inplace=True)
#print(csr_data)

KNN = NearestNeighbors(metric='cosine', algorithm= 'brute', n_neighbors = 20, n_jobs = -1)
KNN.fit(csr_data)

with open('KNN_Model.pkl', 'wb')as file:
    pickle.dump(KNN,file)

