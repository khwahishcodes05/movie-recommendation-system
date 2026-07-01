# =========================================================
# MOVIE RECOMMENDATION SYSTEM (CONTENT-BASED)
# TF-IDF + COSINE SIMILARITY
# =========================================================

# ----------------------------
# IMPORT LIBRARIES
# ----------------------------
import pandas as pd
import ast
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# LOAD DATASET
# ----------------------------
file_path = "movies_metadata.csv"
df = pd.read_csv(file_path)

# ----------------------------
# BASIC DATA CHECK
# ----------------------------
print("Dataset Shape:", df.shape)
print("Missing Values:\n", df.isnull().sum())

# ----------------------------
# DATA CLEANING
# ----------------------------

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Keep only required columns
df = df[["tagline", "overview", "genres", "title"]]

# Handle missing values
df["overview"] = df["overview"].fillna("")
df["tagline"] = df["tagline"].fillna("")
df = df.dropna(subset=["title"])

# Convert genres (list of dict → string)
df["genres"] = df["genres"].apply(
    lambda x: " ".join([i["name"] for i in ast.literal_eval(x)])
)

# Create combined text feature
df["tag"] = df["overview"] + " " + df["genres"] + " " + df["tagline"]

# ----------------------------
# TEXT PREPROCESSING
# ----------------------------

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df["tag"] = df["tag"].apply(clean_text)

# ----------------------------
# INDEX MAPPING
# ----------------------------
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

# ----------------------------
# TF-IDF VECTORIZATION
# ----------------------------
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(df["tag"])

# ----------------------------
# RECOMMENDATION FUNCTION
# ----------------------------
def recommend(movie_title, n=10):
    if movie_title not in indices:
        return "MOVIE NOT FOUND"

    idx = indices[movie_title]

    # similarity calculation
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # top similar movies (excluding itself)
    sim_indices = sim_scores.argsort()[::-1][1:n+1]

    return df["title"].iloc[sim_indices]

# ----------------------------
# TEST
# ----------------------------
print(recommend("Jumanji"))