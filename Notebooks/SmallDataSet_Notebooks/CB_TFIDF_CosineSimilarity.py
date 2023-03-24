import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #obtaining tdf vectors
from itertools import combinations #finding combination of genres for a given movie
from sklearn.metrics.pairwise import cosine_similarity #To compute the cosine similarities between all tf-idf vectors

'''
------------------------------------------------------------------------------------------
✨ Class for Content Based Model: TFIDF+Cosine Similairty✨ 
------------------------------------------------------------------------------------------
'''    
class tfidf_cosine_sim_model():
	def __init__(self,matrix):
		self.matrix=matrix
		self.cosine_sim=cosine_similarity(self.matrix)

	'''
	------------------------------------------------------------------------------------------
	✨ Recommendation Function  ✨ 

	Find the highest tf-idf score for a given movie, function declaration
 
	------------------------------------------------------------------------------------------
	'''
	def genre_recommendation(self, query_title,movies,n=10):
		"""
		Recommends movies based on a similarity dataframe
		Parameters
		----------
		query_title : Movie title (string)
		movies: movies dataframe
		n: Number of recommendations wanted

		"""
		col_name='title'
		cosine_sim_df = pd.DataFrame(self.cosine_sim, index=movies[col_name], columns=movies[col_name])
		items= movies[['movieId','title', 'genres']]
    
		#select column with the input movie title, and change it to numpy array 
		#resulting array of indices indicates the positions of the elements that would be in the first i positions
		sel = cosine_sim_df.loc[:,query_title].to_numpy().argpartition(range(-1,-n,-1)) 
		#resulting subset of column names is ordered in descending order of the corresponding values in the title column. 
		#This subset is then assigned to the variable ct    
		ct = cosine_sim_df.columns[sel[-1:-(n+2):-1]]
		#drop columns title from input and merge the df with the original dataframe. show only first i results. 
		ct = ct.drop(query_title, errors='ignore')
    
		xx = pd.DataFrame(ct).merge(items).head(n)
    
		#add similarity score to xx
		xx['Similarity Score'] = cosine_sim_df.loc[query_title, xx['title']].values
    
		return xx
