# 🎬 Movie Recommendation System (Content-Based)

A content-based movie recommendation system built using Python. It suggests similar movies based on textual features like **overview, genres, and tagline** using **TF-IDF Vectorization** and **Cosine Similarity**.

---

## 🚀 Project Overview

This project analyzes movie metadata and recommends similar movies based on content similarity. It does not use user ratings or collaborative filtering; instead, it relies purely on movie descriptions.

---

## 🧠 How It Works

1. Dataset is loaded and cleaned
2. Important features like `overview`, `genres`, and `tagline` are combined
3. Text preprocessing is applied (lowercasing, stopword removal, lemmatization)
4. TF-IDF Vectorization converts text into numerical form
5. Cosine Similarity measures similarity between movies
6. Top N similar movies are recommended

---

## 📊 Features Used

- Overview
- Genres
- Tagline

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── movie_recommendation_system.py   # Main code file
├── movies_metadata.csv              # Dataset used
├── README.md                        # Project documentation
├── LICENSE                          # Open-source license (MIT recommended)
├── .gitignore                       # Files to ignore in git
```

## 📂 Dataset

Due to GitHub file size limitations, dataset is stored externally.
-------------------------------------------
👉 Download here:  
[Google Drive Link](https://drive.google.com/file/d/1uKwg4FdLQxq7Efv-RRyKJrV4GD6mj2Ch/view?usp=sharing)

