 
import streamlit as st
import pandas as pd
import numpy as np
from scikit-surprise import SVD, Dataset, Reader
from numpy.linalg import norm
import pickle
import os

# --- Load your trained model and data ---
st.title("🎬 Movie Recommender App")

# Paths
DATA_PATH = "data/ml-latest-small"
MODEL_PATH = "notebooks/svd_model.pkl"

# Load data
movies = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
ratings = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))

# Load trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Extract factors
movie_factors = model.qi
movie_to_inner_id = {raw_id: inner_id for raw_id, inner_id in model.trainset._raw2inner_id_items.items()}
inner_id_to_movie = {inner_id: raw_id for raw_id, inner_id in movie_to_inner_id.items()}

# --- Helper functions ---
def get_similar_movies(movie_title, n=10):
    movie_id = movies[movies['title'].str.contains(movie_title, case=False, regex=False)]['movieId']
    if movie_id.empty:
        return []
    movie_id = movie_id.iloc[0]
    inner_id = movie_to_inner_id.get(movie_id)
    if inner_id is None:
        return []
    target_vector = movie_factors[inner_id]
    sims = movie_factors.dot(target_vector) / (norm(movie_factors, axis=1) * norm(target_vector))
    similar_ids = np.argsort(sims)[::-1][1:n+1]
    similar_movie_ids = [inner_id_to_movie[i] for i in similar_ids]
    return movies[movies['movieId'].isin(similar_movie_ids)]['title'].tolist()

def get_user_recommendations(user_id, n=10):
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movies = movies[~movies['movieId'].isin(user_movies)]
    preds = [(mid, model.predict(user_id, mid).est) for mid in all_movies['movieId']]
    top = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    return movies[movies['movieId'].isin([mid for mid, _ in top])]['title'].tolist()

# --- Streamlit UI ---
option = st.sidebar.selectbox("Choose mode:", ["🎥 Similar Movies", "👤 User Recommendations"])

if option == "🎥 Similar Movies":
    st.header("Find Similar Movies")
    movie_name = st.text_input("Enter a movie title:")
    if movie_name:
        recs = get_similar_movies(movie_name)
        if recs:
            st.success("Movies similar to " + movie_name + ":")
            for r in recs:
                st.write("- " + r)
        else:
            st.error("No matches found.")

else:
    st.header("Get User-Based Recommendations")
    user_id = st.number_input("Enter user ID:", min_value=1, step=1)
    if st.button("Recommend"):
        recs = get_user_recommendations(int(user_id))
        if recs:
            st.success(f"Top recommendations for user {user_id}:")
            for r in recs:
                st.write("- " + r)
        else:
            st.error("No recommendations found or user not in training set.")
