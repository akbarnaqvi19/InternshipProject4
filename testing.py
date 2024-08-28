## Ali Akbar Naqvi
## Internship Project 4
## Movie Recommendation System
## Item Based Colaborative Filtering

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix

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

with open('KNN_Model.pkl','rb') as file:
    KNN= pickle.load(file)

def get_recommendation(movies_name):
    movies_list=movies[movies['title'].str.contains(movies_name)]
    if len(movies_list):
        movie_idx = movies_list.iloc[0]['movieId']
        movie_idx= data[data['movieId'] == movie_idx].index[0]
        distance , indices= KNN.kneighbors(csr_data[movie_idx], n_neighbors=11)
        rec_movie_incidences= sorted(list(zip(indices.squeeze().tolist(),distance.squeeze().tolist())), key=lambda  x: x[1])[:0: -1]
        recommended_movies=[]
        for val in rec_movie_incidences:
            movie_idx= data.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommended_movies.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df=pd.DataFrame(recommended_movies, index= range (1,11))
        print("Recommended Movies are :")
        print(df)
        return df
    else:
        print("Movie not Found")

movies_name= input("Enter the Name of a Movie You Have Watched and Liked: ")
get_recommendation(movies_name)