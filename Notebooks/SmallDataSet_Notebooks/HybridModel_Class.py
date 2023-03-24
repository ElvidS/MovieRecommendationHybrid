from collections import defaultdict
from surprise import accuracy
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise.model_selection import train_test_split,cross_validate,RandomizedSearchCV
from surprise import KNNWithMeans
from surprise import KNNBasic
import pandas as pd
import numpy as np

'''
------------------------------------------------------------------------------------------
✨ Class for Hybrid Model: Collaborative Filtering + Content Based ✨ 
------------------------------------------------------------------------------------------
'''    
class hybrid_model():
	def __init__(self,knn_model,tfidf_cosinesim_model):
		self.knn_model = knn_model
		self.tfidf_cosinesim_model = tfidf_cosinesim_model
	
	def extract_full_cosinesimilarities(self,user_input,movies):
		# movies: movies dataframe, no user info 
		#Number of unique movies
		n_unique_movies=len(movies.movieId.unique())

		similar_movies_cos_sim=self.tfidf_cosinesim_model.genre_recommendation(user_input,movies,n_unique_movies)	
		return similar_movies_cos_sim
		
	'''
	------------------------------------------------------------------------------------------
	✨ Recommendation Function  ✨ 
	------------------------------------------------------------------------------------------
	'''
		
	def recommend_similar_items_hybrid(self,user_input, df, n=10):
	
		#-------------------------------------------------
		# Content Based
		#-------------------------------------------------
 
		# Simplify to use in content based model
		df_copy=df.copy()
		df_copy['pasteIDandMovie'] = df_copy['title']+str(df_copy['movieId'])
		df_copy = df_copy.drop_duplicates(subset=['pasteIDandMovie'])
		movies = df_copy[['movieId', 'title', 'genres']].sort_values(by=['movieId']).reset_index(drop=True)
	
		# We will extract the all cosine similarities from the content based model
		similar_movies_cos_sim=self.extract_full_cosinesimilarities(user_input,movies)
	
		#--------------------------------------
		# Col. filter Based
		#--------------------------------------
	
		# Find the top n*10 similar movies based on the Coll filter model
		similar_movies_knn=self.knn_model.recommend_similar_items_knnmeans(user_input,df,n*10)
	 
		# Find the cosine similarity scores for all these movies
		
		similar_movies_knn_cos_sim=pd.merge(similar_movies_knn,similar_movies_cos_sim,how='left', on=['movieId','title','genres'])
		hybrid_recommendations=similar_movies_knn_cos_sim.sort_values(['Similarity Score','average rating','number of ratings'], ascending=[False,False,False]).head(n)

		#Rename Similarity Score
		hybrid_recommendations=hybrid_recommendations.rename(columns={"Similarity Score":"Genre Similarity Score"})
		return similar_movies_cos_sim.head(n), similar_movies_knn, hybrid_recommendations

		

