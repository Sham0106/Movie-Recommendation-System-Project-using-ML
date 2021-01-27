#!/usr/bin/env python
# coding: utf-8

# # Movie recommendation system

# # Index
# - Exploratory Data Analysis(EDA)
# - Collaborative Filtering
#     - Memory based collaborative filtering
#         - User-Item Filtering
#         - Item-Item Filtering

# ## Load Libraries 

# In[350]:


from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# # Dataset : Movielens
# https://grouplens.org/datasets/movielens/100k

# In[351]:


# Reading ratings file
ratings = pd.read_csv('ratings.csv', sep=',', encoding='latin-1', usecols=['userId','movieId','rating','timestamp'])

# Reading movies file
movies = pd.read_csv('movies.csv', sep=',', encoding='latin-1', usecols=['movieId','title','genres'])


# In[352]:


df_movies = movies 
df_ratings = ratings 


# ## Exploratory Data Analysis(EDA)

# In[353]:


df_movies.head(5)


# ### Most popular genres of movie released

# In[354]:


plt.figure(figsize=(20,7))
generlist = df_movies['genres'].apply(lambda generlist_movie : str(generlist_movie).split("|"))
geners_count = {}

for generlist_movie in generlist:
    for gener in generlist_movie:
        if(geners_count.get(gener,False)):
            geners_count[gener]=geners_count[gener]+1
        else:
            geners_count[gener] = 1       
geners_count.pop("(no genres listed)")
plt.bar(geners_count.keys(),geners_count.values(),color='m')


# In[355]:


df_ratings.head(5)


# ### Distribution of users rating

# In[356]:


sns.distplot(df_ratings["rating"]);


# In[357]:


print("Shape of frames: \n"+ " Rating DataFrame"+ str(df_ratings.shape)+"\n Movies DataFrame"+ str(df_movies.shape))


# In[358]:


merge_ratings_movies = pd.merge(df_movies, df_ratings, on='movieId', how='inner')


# In[359]:


merge_ratings_movies.head(2)


# In[360]:


merge_ratings_movies = merge_ratings_movies.drop('timestamp', axis=1)


# In[361]:


merge_ratings_movies.shape


# Grouping the rating based on user

# In[362]:


ratings_grouped_by_users = merge_ratings_movies.groupby('userId').agg([np.size, np.mean])


# In[363]:


ratings_grouped_by_users.head(2)


# In[364]:


ratings_grouped_by_users = ratings_grouped_by_users.drop('movieId', axis = 1)


# ### Top 10 users who have rated most of the movies

# In[365]:


ratings_grouped_by_users['rating']['size'].sort_values(ascending=False).head(10).plot('bar', figsize = (10,5))


# In[366]:


ratings_grouped_by_movies = merge_ratings_movies.groupby('movieId').agg([np.mean], np.size)


# In[367]:


ratings_grouped_by_movies.shape


# In[368]:


ratings_grouped_by_movies.head(3)


# In[369]:


ratings_grouped_by_movies = ratings_grouped_by_movies.drop('userId', axis=1)


# ### Movies with high average rating

# In[370]:


ratings_grouped_by_movies['rating']['mean'].sort_values(ascending=False).head(10).plot(kind='barh', figsize=(7,6));


# ### Movies with low average rating

# In[371]:


low_rated_movies_filter = ratings_grouped_by_movies['rating']['mean']< 1.5


# In[372]:


low_rated_movies = ratings_grouped_by_movies[low_rated_movies_filter]


# In[373]:


low_rated_movies.head(20).plot(kind='barh', figsize=(7,5));


# In[374]:


low_rated_movies.head(10)


# # Collaborative Filtering
# Types of collaborative filtering techniques
# * Memory based 
#  - User-Item Filtering
#  - Item-Item Filtering
# * Model based 
#  - Matrix Factorization
#  - Clustering
#  - Deep Learning
# 

# ## Implementation of Item-Item Filtering

# In[383]:


df_movies_ratings=pd.merge(df_movies, df_ratings)


# In[384]:


df_movies_ratings


# Here Pivot table function is used as we want one to one maping between movies, user and their rating. 
# So by default pivot_table command takes average if we have multiple values of one combination.

# In[385]:


ratings_matrix_items = df_movies_ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix_items.fillna( 0, inplace = True )
ratings_matrix_items.shape


# In[386]:


ratings_matrix_items


# In[387]:


movie_similarity = 1 - pairwise_distances( ratings_matrix_items.as_matrix(), metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix_items = pd.DataFrame( movie_similarity )
ratings_matrix_items


# Below function will take the movie name as a input and will find the movies which are similar to this movie.
# This function first find the index of movie in movies frame and then take the similarity of movie and align in movies dataframe so that we can get the similarity of the movie with all other movies.

# In[388]:


def item_similarity(movieName): 
    """
    recomendates similar movies
   :param data: name of the movie 
   """
    try:
        #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
        user_inp=movieName
        inp=df_movies[df_movies['title']==user_inp].index.tolist()
        inp=inp[0]

        df_movies['similarity'] = ratings_matrix_items.iloc[inp]
        df_movies.columns = ['movie_id', 'title', 'release_date','similarity']
    except:
        print("Sorry, the movie is not in the database!")


# Here we provide the user id of the user for which we have to recommend movies.
# Then we find the movies which are rated 5 or 4.5 by the user for whom we want to recommend movies.
# We are finding this because as we know that in Item-Item similarity approach we recommended movies to the user based on his previous selection.
# So to foster our algorithm we are finding movies which are liked by the user most and on bases of that we will recommend movies with are similar to movies highly rated by the user.
# Then our function has appended the similarity of the movie highly rated by the user to our movies data frame.
# Now we will sort the frame as per the similarity in descending order so that we can get the movies which are highly similar to movie highly rated bu our customer.
# Now we filter the movies which are most similar as per the similarity so if similarity is greater than 0.45 then we are considering the movies.
# Now the function goes ahead and see which all movies user has seen and then filter out the movies which he has not seen and than recommended that movies to him.

# In[389]:


def recommendedMoviesAsperItemSimilarity(user_id):
    """
     Recommending movie which user hasn't watched as per Item Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user_movie= df_movies_ratings[(df_movies_ratings.userId==user_id) & df_movies_ratings.rating.isin([5,4.5])][['title']]
    user_movie=user_movie.iloc[0,0]
    item_similarity(user_movie)
    sorted_movies_as_per_userChoice=df_movies.sort_values( ["similarity"], ascending = False )
    sorted_movies_as_per_userChoice=sorted_movies_as_per_userChoice[sorted_movies_as_per_userChoice['similarity'] >=0.45]['movie_id']
    recommended_movies=list()
    df_recommended_item=pd.DataFrame()
    user2Movies= df_ratings[df_ratings['userId']== user_id]['movieId']
    for movieId in sorted_movies_as_per_userChoice:
            if movieId not in user2Movies:
                df_new= df_ratings[(df_ratings.movieId==movieId)]
                df_recommended_item=pd.concat([df_recommended_item,df_new])
            best10=df_recommended_item.sort_values(["rating"], ascending = False )[1:10] 
    return best10['movieId']


# In[390]:


def movieIdToTitle(listMovieIDs):
    """
     Converting movieId to titles
    :param user_id: List of movies
    :return: movie titles
    """
    movie_titles= list()
    for id in listMovieIDs:
        movie_titles.append(df_movies[df_movies['movie_id']==id]['title'])
    return movie_titles


# In[391]:


user_id=50
print("Recommended movies,:\n",movieIdToTitle(recommendedMoviesAsperItemSimilarity(user_id)))


# ## Implementation of User-Item Filtering

# In similar way as we did for ItemItem similarity we will create a matrix but here we will keep rows as user and columns as movieId as we want a vector of different users.
# Then in similar ways we will find distance and similarity between users.

# In[392]:


ratings_matrix_users = df_movies_ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').reset_index(drop=True)
ratings_matrix_users.fillna( 0, inplace = True )
movie_similarity = 1 - pairwise_distances( ratings_matrix_users.as_matrix(), metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix_users = pd.DataFrame( movie_similarity )
ratings_matrix_users


# Here now we have similarity of users in colums with respective users in row. So if we find maximum value in a column we will get the user with highest similarity. So now we can have a pair of users which are similar.

# In[393]:


ratings_matrix_users.idxmax(axis=1)


# In[394]:


ratings_matrix_users.idxmax(axis=1).sample( 10, random_state = 10 )


# In[395]:


similar_user_series= ratings_matrix_users.idxmax(axis=1)
df_similar_user= similar_user_series.to_frame()


# In[396]:


df_similar_user.columns=['similarUser']


# In[397]:


df_similar_user


# Below function takes id of the user to whom we have to recommend movies. On basis of that, we find the user which is similar to that user and then filter the movies which are highly rated by the user to recommend them to given user.

# In[398]:


movieId_recommended=list()
def getRecommendedMoviesAsperUserSimilarity(userId):
    """
     Recommending movies which user hasn't watched as per User Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user2Movies= df_ratings[df_ratings['userId']== userId]['movieId']
    sim_user=df_similar_user.iloc[0,0]
    df_recommended=pd.DataFrame(columns=['movieId','title','genres','userId','rating','timestamp'])
    for movieId in df_ratings[df_ratings['userId']== sim_user]['movieId']:
        if movieId not in user2Movies:
            df_new= df_movies_ratings[(df_movies_ratings.userId==sim_user) & (df_movies_ratings.movieId==movieId)]
            df_recommended=pd.concat([df_recommended,df_new])
        best10=df_recommended.sort_values(['rating'], ascending = False )[1:10]  
    return best10['movieId']


# In[399]:


user_id=50
recommend_movies= movieIdToTitle(getRecommendedMoviesAsperUserSimilarity(user_id))
print("Movies you should watch are:\n")
print(recommend_movies)


# # Evaluating the model

# In[403]:


def get_user_similar_movies( user1, user2 ):
    
    """
     Returning common movies and ratings of same for both the users
    :param user1,user2: user ids of 2 users need to compare
    :return: movieIds to user 
    """
    common_movies = df_movies_ratings[df_movies_ratings.userId == user1].merge(
      df_movies_ratings[df_movies_ratings.userId == user2],
      on = "movieId",
      how = "inner" )
    common_movies.drop(['movieId','genres_x','genres_y', 'timestamp_x','timestamp_y','title_y'],axis=1,inplace=True)
    return common_movies


# In[404]:


get_user_similar_movies(587,511)


# ## Pros and Cons of two methods
# 
# Challenges with User similarity
# - The challenge with calculating user similarity is the user need to have some prior purchases and should have rated them.
# - This recommendation technique does not work for new users.
# - The system need to wait until the user make some purchases and rates them. Only then similar users can be found and recommendations can be made. This is called cold start problem.

# # Evaluating Collaborative Filtering

# ***Hit Ratio***
# It is ratio of number of hits/ Total recommendation

# In[436]:


user_id=50


# In[437]:


def evaluation_collaborative_svd_model(userId,userOrItem):
    """
    hydrid the functionality of Collaborative based and svd based model to see if ratings of predicted movies 
    :param userId: userId of user, userOrItem is a boolean value if True it is User-User and if false Item-Item
    :return: dataframe of movies and ratings
    """ 
    movieIdsList= list()
    movieRatingList=list()
    movieIdRating= pd.DataFrame(columns=['movieId','rating'])
    if userOrItem== True:
        movieIdsList=getRecommendedMoviesAsperUserSimilarity(userId)
    else:
        movieIdsList=recommendedMoviesAsperItemSimilarity(user_id)
    for movieId in movieIdsList:
        predict = svd.predict(userId, movieId)
        movieRatingList.append([movieId,predict.est])
        movieIdRating = pd.DataFrame(np.array(movieRatingList), columns=['movieId','rating'])
        count=movieIdRating[(movieIdRating['rating'])>=3]['movieId'].count()
        total=movieIdRating.shape[0]
        hit_ratio= count/total
    return hit_ratio
    


# In[438]:


print("Hit ratio of User-user collaborative filtering")
print(evaluation_collaborative_svd_model(user_id,True))
print("Hit ratio of Item-Item collaborative filtering")
print(evaluation_collaborative_svd_model(user_id,False))

