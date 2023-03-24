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
