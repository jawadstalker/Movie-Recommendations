# Movie Recommendations

A semantic movie recommender that matches a natural-language description to films in the TMDB 5000 dataset. It uses Sentence Transformers embeddings and cosine similarity, with a lightweight Tkinter desktop interface.

## Features

- Natural-language search rather than exact keyword matching
- Semantic representations built from title, tagline, overview, genres, cast, and keywords
- Cached, normalized embeddings — the transformer model does not re-encode the full catalogue on every launch
- Optional release-year and genre filters in the Python recommendation API
- Configurable result count and poster previews
- Clear validation for missing or malformed dataset files

## Requirements

- Python 3.9+
- The two CSV files from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Setup

```bash
git clone https://github.com/jawadstalker/Movie-Recommendations.git
cd Movie-Recommendations
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from Kaggle and place both files in the project root. The CSV files are not included because of their size and licensing.

Run the desktop app:

```bash
python movie.py
```

The first launch downloads the sentence-transformer model and computes movie embeddings. This can take a while depending on your hardware. Later launches reuse `.cache/movie_embeddings.npz`. Delete that cache file to force a rebuild.

## Python API

The recommendation pipeline can also be used without the desktop UI:

```python
from sentence_transformers import SentenceTransformer
from recommender import (
    MODEL_NAME,
    build_movie_texts,
    get_movie_embeddings,
    load_movies,
    recommend_movies,
)

movies = load_movies("data")
model = SentenceTransformer(MODEL_NAME)
texts = build_movie_texts(movies)
embeddings = get_movie_embeddings(texts, model)

results = recommend_movies(
    "A thoughtful science-fiction film about memory and identity",
    movies,
    embeddings,
    model,
    top_k=10,
)
print(results[["title", "release_date", "similarity"]])
```

The API supports `min_year`, `max_year`, and `genre` filters. The GUI currently exposes the query and result count.

## Project structure

```text
.
├── movie.py                 # Tkinter desktop application
├── recommender.py           # Data preparation, embedding cache, ranking
├── tests/
│   └── test_recommender.py  # Unit tests
├── requirements.txt
└── README.md
```

## Model and limitations

The default model is `sentence-transformers/all-MiniLM-L6-v2`, selected as a smaller and faster alternative to larger transformer encoders. It is suitable for a local prototype, though recommendation quality should be evaluated against a labelled or user-rated test set before treating rankings as objective.

Similarity is semantic relevance, not a probability that a user will like a film. Poster images are fetched from TMDB's image CDN when available; poster loading requires an internet connection.

## Tests

```bash
python -m pytest
```

## Data and attribution

- Movie metadata: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Embedding model: [Sentence Transformers](https://www.sbert.net/)
- Poster images: [The Movie Database (TMDB)](https://www.themoviedb.org/)

This is an educational project and is not affiliated with TMDB.

## Hybrid recommendations (optional MovieLens data)

The optional hybrid engine combines semantic content relevance with collaborative signals from MovieLens user ratings. Without a known user and the MovieLens files, it falls back to content-based recommendations.

Download a MovieLens dataset containing `ratings.csv` and `links.csv` and place both files in the project root. The links file maps MovieLens movie IDs to TMDB IDs, so ratings can be aligned with the TMDB 5000 catalogue.

```python
from sentence_transformers import SentenceTransformer
from recommender import MODEL_NAME, build_movie_texts, get_movie_embeddings, load_movies
from hybrid import load_movielens_data, build_collaborative_model, recommend_hybrid

movies = load_movies()
model = SentenceTransformer(MODEL_NAME)
embeddings = get_movie_embeddings(build_movie_texts(movies), model)
ratings, links = load_movielens_data()
collaborative = build_collaborative_model(ratings, links, movies)

results = recommend_hybrid(
    "A mystery thriller with an unreliable narrator",
    movies, embeddings, model,
    user_id=1, collaborative_model=collaborative, top_k=10,
    content_weight=0.65,
)
print(results[["title", "similarity", "collaborative_score", "hybrid_score"]])
```

The default blend assigns 65% weight to semantic relevance and 35% to collaborative preference when a collaborative prediction exists. Movies without collaborative evidence keep their content score. Unknown users automatically use the content-based cold-start path. MovieLens files are optional and are not committed to the repository.
