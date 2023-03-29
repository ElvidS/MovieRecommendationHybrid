import streamlit as st
import pandas as pd
import streamlit as st
#from HybridModel_Class import hybrid_model
from CFKnnMeansModel_Class import collab_filtering_Kmeans_Model
#from CB_TFIDF_CosineSimilarity import tfidf_cosine_sim_model
#copy-paste functions, from all modelss. 
import pickle
import utilities


#FUNCTIONS
## 1. Genre Recommendation from content based notebook

def genre_recommendation(query_title):
    """
    Recommends movies based on a similarity dataframe
    Parameters
    ----------
    query_title : Movie title (string)

    """
    items= movies[['title', 'genres']]
    #select column with the input movie title, and change it to numpy array 
    #resulting array of indices indicates the positions of the elements that would be in the first i positions
    sel = cosine_sim_df.loc[:,query_title].to_numpy().argpartition(range(-1,-10,-1)) 
    #resulting subset of column names is ordered in descending order of the corresponding values in the title column. 
    #This subset is then assigned to the variable ct    
    ct = cosine_sim_df.columns[sel[-1:-(10+2):-1]]
    #drop columns title from input and merge the df with the original dataframe. show only first i results. 
    ct = ct.drop(query_title, errors='ignore')
    
    xx = pd.DataFrame(ct).merge(items).head(10)
    
    #add similarity score to xx
    xx['Similarity Score'] = cosine_sim_df.loc[query_title, xx['title']].values
    
    return xx

## 2. Hybrid model class from Naive Hybrid notebook
class HybridModel:
    def __init__(self, cosine_sim, cf_model):
        self.cosine_sim = cosine_sim
        self.cf_model = cf_model
        
    def recommend_movies(self, user_title_year, movies_df):
        
        # Use the Process_Avg_Rating function to manipulate the main df and find the 
        # avg rating
        
        movies_df_summary=Process_Avg_Rating(movies_df)
        
        #--------------------------------------
        # Content Based
        #--------------------------------------
        
        # Find the top 100 similar movies based on the content-based model
        similar_movies_cos_sim=genre_recommendation(user_title_year)
        
          
        #Merge
        similar_movies_cos_sim_df=pd.merge(similar_movies_cos_sim,movies_df_summary,how='left', left_on=['title','genres'], right_on = ['title','genres'])
        
        #--------------------------------------
        # Col. filter Based
        #--------------------------------------
        
        # Find the top 100 similar movies based on the Coll filter model
        similar_movies_knn=self.cf_model.recommend_similar_items_knnmeans(user_input,movies_df,100)
        
        #Take the common 10 movies
       
        similar_movies_common=pd.merge(similar_movies_knn,similar_movies_cos_sim_df, how='inner', on=['title','average rating','number of ratings'])
        similar_movies_common=similar_movies_common.sort_values(['average rating'], ascending=[False]).head(10)

        #What if most common movies are <10?
        
        return similar_movies_cos_sim_df, similar_movies_knn,similar_movies_common
    
    
## 3. Process_Avg_Rating 

def Process_Avg_Rating(inp_df):
    df_out_0=inp_df.drop(["userId"],axis=1).groupby(['movieId','title',"year","genres"])
    df_out=df_out_0.mean()
    df_out['average rating']=df_out['rating'].round(2)
    df_out=df_out.drop(['rating'],axis=1)
    df_out['number of ratings']= df_out_0['title'].count()
    return df_out 

## movie list definition
def MovieList():
    df=pd.read_csv("MovieRecommendationHybrid/Data/ml-latest-small/PreprocessedData_ml_latest_year_small.csv",index_col=0)
    movies=df["title"].unique()
    return movies

### end of functions
hybrid_model = pickle.load(open('MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_hybrid.sav', 'rb'))

#BACKGROUND
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(https://gallery.yopriceville.com/var/albums/Backgrounds/Cinema_Background.jpg);
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
#end of back ground

#TITEL
# match title color with carpet light and move it a bit 
st.markdown(
    """
    <style>
    /* CSS to move the title up */
    .title {
        margin-top: -100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style=color:rgb(205,107,136)>Movie recommendation system</h1>", unsafe_allow_html=True)
#### END of Title

#DROPDOWN. enable only one selection
movies =MovieList()
options = st.multiselect(label="Select Movie", options=movies, max_selections=2)
if options:
    st.write(f"You selected {len(options)} movie:")
    for movie in options:
        st.write(f"- {movie}")
else:
    st.write("Please select one movie")

### Movierecommendation button###

def run_calculation():
    # Call the function from the other notebook
    #knn_model = pickle.load(open('MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_KNN_Means.sav', 'rb'))
    #cos_sim_model = pickle.load(open('MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_tfidf_cosine_sim.sav', 'rb'))
    #HybridModel=hybrid_model(knn_model,cos_sim_model)
    df=pd.read_csv("MovieRecommendationHybrid/Data/ml-latest-small/PreprocessedData_ml_latest_year_small.csv",index_col=0)
    content_df,collab_df,hybrid_df=hybrid_model.recommend_movies(options,df) #options was "user_input"
    # Display the result in Streamlit
    st.write("The result is:", hybrid_df.head(n=10))
#import pickle
#hybrid_filename = "MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_hybrid.sav"
#hybrid_model = pickle.load(open(hybrid_filename, 'rb'))
# Create a button that calls the run_calculation function when clicked
if st.button("Go"):
    run_calculation()
