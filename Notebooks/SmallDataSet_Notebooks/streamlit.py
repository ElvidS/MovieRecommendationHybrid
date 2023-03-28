import streamlit as st
import pandas as pd
import streamlit as st
from HybridModel_Class import hybrid_model
from CFKnnMeansModel_Class import collab_filtering_Kmeans_Model
from CB_TFIDF_CosineSimilarity import tfidf_cosine_sim_model
import pickle


#FUNCTION definition
def MovieList():
    df=pd.read_csv("MovieRecommendationHybrid/Data/ml-latest-small/PreprocessedData_ml_latest_year_small.csv",index_col=0)
    movies=df["title"].unique()
    return movies


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
    knn_model = pickle.load(open('MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_KNN_Means.sav', 'rb'))
    cos_sim_model = pickle.load(open('MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_tfidf_cosine_sim.sav', 'rb'))
    HybridModel=hybrid_model(knn_model,cos_sim_model)
    df=pd.read_csv("MovieRecommendationHybrid/Data/ml-latest-small/PreprocessedData_ml_latest_year_small.csv",index_col=0)
    content_df,collab_df,hybrid_df=HybridModel.recommend_similar_items_hybrid(options,df,10) #options was "user_input"
    # Display the result in Streamlit
    st.write("The result is:", hybrid_df.head(n=10))
#import pickle
#hybrid_filename = "MovieRecommendationHybrid/Notebooks/SmallDataSet_Notebooks/Model_hybrid.sav"
#hybrid_model = pickle.load(open(hybrid_filename, 'rb'))
# Create a button that calls the run_calculation function when clicked
if st.button("Go"):
    run_calculation()
