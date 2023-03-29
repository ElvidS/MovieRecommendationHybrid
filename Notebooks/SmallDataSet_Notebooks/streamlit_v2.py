import streamlit as st
import pandas as pd
import streamlit as st
#from HybridModel_Class import hybrid_model
from CFKnnMeansModel_Class import collab_filtering_Kmeans_Model, Process_Avg_Rating
#from CB_TFIDF_CosineSimilarity import tfidf_cosine_sim_model
# copy-paste functions, from all modelss.
import pickle


# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------


# 1. Genre Recommendation from content based notebook

def genre_recommendation(query_title):
    """
    Recommends movies based on a similarity dataframe
    Parameters
    ----------
    query_title : Movie title (string)

    """
    items = movies[['title', 'genres']]
    # select column with the input movie title, and change it to numpy array
    # resulting array of indices indicates the positions of the elements that would be in the first i positions
    sel = cosine_sim_df.loc[:, query_title].to_numpy().argpartition(range(-1, -100, -1))
    # resulting subset of column names is ordered in descending order of the corresponding values in the title column.
    # This subset is then assigned to the variable ct
    ct = cosine_sim_df.columns[sel[-1:-(100+2):-1]]
    # drop columns title from input and merge the df with the original dataframe. show only first i results.
    ct = ct.drop(query_title, errors='ignore')

    xx = pd.DataFrame(ct).merge(items).head(100)

    # add similarity score to xx
    xx['Similarity Score'] = cosine_sim_df.loc[query_title, xx['title']].values

    return xx


# 3. Hybrid model class from Naive Hybrid notebook
class HybridModel:
    def __init__(self, cosine_sim, cf_model):
        self.cosine_sim = cosine_sim
        self.cf_model = cf_model
    def recommend_movies(self, user_title_year, movies_df):
        # Use the Process_Avg_Rating function to manipulate the main df and find the avg rating
        movies_df_summary = Process_Avg_Rating(movies_df)
        
        # --------------------------------------
        # Content Based
        # --------------------------------------
        # Find the top 100 similar movies based on the content-based model
        similar_movies_cos_sim = genre_recommendation(user_title_year)

		# Merge
        similar_movies_cos_sim_df = pd.merge(similar_movies_cos_sim, movies_df_summary, how='left', left_on=['title', 'genres'], right_on=['title', 'genres'])

		# --------------------------------------
		# Col. filter Based
		# --------------------------------------

		# Find the top 100 similar movies based on the Coll filter model
        similar_movies_knn = self.cf_model.recommend_similar_items_knnmeans(user_title_year, movies_df, 100)

		# Take the common 10 movies

        similar_movies_common = pd.merge(similar_movies_knn, similar_movies_cos_sim_df, how='inner', on=['title', 'average rating', 'number of ratings'])
        similar_movies_common = similar_movies_common.sort_values(['average rating'], ascending=[False]).head(10)

		# What if most common movies are <10?
 
        return similar_movies_cos_sim_df, similar_movies_knn, similar_movies_common


# 4. movie list definition


def MovieList(df):
    movies = df["title"].unique()
    return movies

# 5. BACKGROUND


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
# 6. Hybrid Recommendation Streamlit function


def run_calculation(df, hybrid_model, user_fav_movie):

    content_df, collab_df, hybrid_df = hybrid_model.recommend_movies(user_fav_movie, df)
    # Display the result in Streamlit
    st.write("The result is:", hybrid_df.head(n=10))

# end of functions
# -----------------------------------------------


# -----------------------------------------------
# Main Streamlit script
# -----------------------------------------------

# Background
add_bg_from_url()
# end of back ground

# TITEL
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

st.markdown("<h1 style=color:rgb(205,107,136)>Movie recommendation system</h1>",
            unsafe_allow_html=True)
# END of Title

# Read The dataframe
df = pd.read_csv("MovieRecommendationHybrid/Data/ml-latest-small/PreprocessedData_ml_latest_year_small.csv")
# This part is taken from Femke's notebook
# The way it is processed there is somehow different so put a copy of the dataframe to do that (quick fix)
# If not do this, the number of ratings at the end of the run calculations is wrong
df_copy=df.copy()
df_copy['pasteIDandMovie'] = df_copy['title']+str(df_copy['movieId'])
df_copy = df_copy.drop_duplicates(subset=['pasteIDandMovie'])
movies = df_copy[['movieId', 'title', 'genres']].sort_values(by=['movieId']).reset_index(drop=True)


# DROPDOWN. enable only one selection
movie_List = MovieList(df)
options = st.multiselect(label="Select a Movie of your liking", options=movie_List, max_selections=1)
if options:
    st.write(f"You selected {len(options)} movie:")
    for movie in options:
        st.write(f"- {movie}")
else:
    st.write("Please select one movie")

# Load Hybrid Model
hybrid_model = pickle.load(open('MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_hybrid.sav', 'rb'))
# cosine sim df definition
cosine_sim_df = pd.DataFrame(hybrid_model.cosine_sim, index=movies['title'], columns=movies['title'])
### Movierecommendation button###
# Create a button that calls the run_calculation function when clicked
if st.button("Go"):
    # print(options[0])
    run_calculation(df, hybrid_model, options[0])
