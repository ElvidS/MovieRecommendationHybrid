import streamlit as st
import pandas as pd
#background

def MovieList():
    df=pd.read_csv("../../Data/ml-latest-small/PreprocessedData_ml_latest_year_small.csv",index_col=0)
    movies=df["title"].unique()
    return movies

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

st.sidebar.header("Options")

st.title("Movie recommendation system")
button1 = st.button("click me")
if button1:
    st.write("this is some text")
st.header("Start of checkbox section")
like = st.checkbox("Do you like this app?")
button2= st.button("Submit")
if button2:
    if like:
        st.write("thanks. I like it as well")
    else:
        st.write("im sorry, you have bad tastes")
st.header("Movies")
user_text = st.text_input("What is your favourite movie?")
if st.button("Save"):
    st.write(user_text)

#movies = ["Elvid, the movie star", "Athoy, the movie star", "Femke, upcoming movie star"]
movies =MovieList()
options = st.multiselect(label="Select Movies", options=movies)

st.text("")
st.text("")
