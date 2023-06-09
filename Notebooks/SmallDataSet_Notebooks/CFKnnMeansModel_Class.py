from collections import defaultdict
from surprise import accuracy
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise.model_selection import train_test_split,cross_validate,RandomizedSearchCV
from surprise import KNNWithMeans
from surprise import KNNBasic
import pandas as pd
import numpy as np

'''✨ Functions that will be used many times in streamlit and other models ✨ '''

def AskForUserInput(df):
    fav_movie=input("Enter your Favorite Movie: ").lower()
    n=0
    
    movies=df[df['title'].str.lower().str.contains(fav_movie)].drop(['userId','rating','genres'],axis=1).drop_duplicates()
    
    #upper case dependency removed
    #year removed
    
    if movies.shape[0]==1:
        print("We have your favourite movie in our database!")
        return fav_movie
    elif movies.shape[0]>1:
        print("\nWe have multiple movies with the same name/Part of it, but with different release years:")
        print(movies.to_string(index=False))
        
        fav_movie_id=int(input("Which one do you have in your mind? (Enter the movieId)"))
        ids=movies["movieId"].unique()
        if fav_movie_id not in ids :
            print("Wrong id! Taking the first one")
            fav_movie=movies.iloc[0]['title']
            #print(fav_movie)
        else:
            fav_movie=movies[movies['movieId']==fav_movie_id].iloc[0]['title']
       
    else:
        print("Unfortunately, We do not have your favourite movie in our list.")
        fav_movie="None"
    
    print("Your favourite movie:",fav_movie)
    return fav_movie

def Process_Avg_Rating(inp_df):
    df_out_0=inp_df.drop(["userId"],axis=1).groupby(['movieId','title',"year","genres"])
    df_out=df_out_0.mean()
    df_out['average rating']=df_out['rating'].round(2)
    df_out=df_out.drop(['rating'],axis=1)
    df_out['number of ratings']= df_out_0['title'].count()
    return df_out 

'''
------------------------------------------------------------------------------------------
✨ Class for Collaborative Filtering with KNN with Means  ✨ 
------------------------------------------------------------------------------------------
'''    
class collab_filtering_Kmeans_Model():
	def __init__(self, model,trainset, testset):
		self.model = model
		self.trainset = trainset
		self.testset = testset
	def fit_and_predict(self,df,n):
		print('**Fitting the train data...**')
		self.model.fit(self.trainset)
		print('**Predicting the test data...**')
		pred_test = self.model.test(self.testset)
		rmse = round(accuracy.rmse(pred_test), 3)
		print('**RMSE for the predicted result is ' + str(rmse) + '**')
		
		#Top n
		# First map the predictions to each user.
		top_n = defaultdict(list)
		for uid, iid, true_r, est, _ in pred_test:
			#cross-relate other information from the fulldf
			movieName=df[df['movieId']==iid]['title'].unique()[0]
			movieYear=df[df['movieId']==iid]['year'].unique()[0]
			genres=df[df['movieId']==iid]['genres'].unique()[0]
			avgRat=df[df['movieId']==iid]['rating'].mean().round(2)
			ratedBy=len(df[df['movieId']==iid]['rating'])
			top_n[uid].append((iid, movieName, movieYear, genres, avgRat, ratedBy))
		
		# Then sort the predictions for each user and retrieve the k highest ones.
		for uid, user_ratings in top_n.items():
			user_ratings.sort(key=lambda x: x[1], reverse=True)
			top_n[uid] = user_ratings[:n]
			
		recommenddf = pd.DataFrame(columns=['userId', 'movieId', 'title', 'year', 'genres', 'average rating','number of ratings'])
		for item in top_n:
			subdf = pd.DataFrame(top_n[item], columns=['movieId','title',  'year', 'genres', 'average rating','number of ratings'])
			subdf['userId'] = item
			cols = subdf.columns.tolist()
			cols = cols[-1:] + cols[:-1]
			subdf = subdf[cols]
			recommenddf = pd.concat([recommenddf, subdf], axis = 0)
		
		return rmse, recommenddf

	def cross_validate(self,data):
		print('**Cross Validating the data...**')
		cv_result = cross_validate(self.model, data, n_jobs=-1,cv=5,verbose = True)
		return cv_result

	'''
	------------------------------------------------------------------------------------------
	✨ Recommendation Function  ✨ 

	Users and items have a raw id and an inner id. Some methods will use/return a raw id (e.g. 
	the predict() method), while some other will use/return an inner id.

	Raw ids are ids as defined in a rating file or in a pandas dataframe. They can be strings 
	or numbers. Note though that if the ratings were read from a file which is the standard 
	scenario, they are represented as strings. This is important to know if you’re using 
	e.g. predict() or other methods that accept raw ids as parameters.

	On trainset creation, each raw id is mapped to a unique integer called inner id, which is 
	a lot more suitable for Surprise to manipulate. Conversions between raw and inner ids can 
	be done using the to_inner_uid(), to_inner_iid(), to_raw_uid(), and to_raw_iid() methods 
	of the trainset.

	------------------------------------------------------------------------------------------
	'''
	def recommend_similar_items_knnmeans(self,movie_title, df, n=5):
		# Take the first occurrance of the movie
		model=self.model 
		movieId=df[df['title']==movie_title]['movieId'].unique()[0]
		inner_movieId=model.trainset.to_inner_iid(movieId)
		movie_neighbours=model.get_neighbors(inner_movieId,n)
		df_out=df[df.movieId.isin([model.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbours])]
		df_out=Process_Avg_Rating(df_out)
		return df_out 
