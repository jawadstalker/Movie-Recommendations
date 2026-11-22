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

Download the MovieLens **latest-small** dataset (recommended for this local prototype), which contains `ratings.csv` and `links.csv`, and place both files in the project root. Avoid the full MovieLens dataset on low-memory machines because its ratings file is substantially larger. The links file maps MovieLens movie IDs to TMDB IDs, so ratings can be aligned with the TMDB 5000 catalogue.

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

## Recommendation evaluation

Use the evaluation runner to compare Content-Based and Hybrid ranking with a per-user temporal holdout. The latest interaction for each eligible user is held out; earlier interactions are used to build the collaborative model and the user's preference profile. This prevents the held-out interaction from leaking into training.

Install the project requirements and run:

```bash
python evaluate.py --k 10 --threshold 4.0
```

Optional arguments:

- `--k 10`: number of recommendations evaluated per user
- `--threshold 4.0`: ratings at or above this value count as relevant
- `--min-train-ratings 3`: minimum historical interactions required before the held-out event
- `--content-weight 0.65`: semantic-content weight in the hybrid score
- `--max-users 100`: optional smaller run for a quick smoke test

The runner reports Precision@K, Recall@K, MAP@K, NDCG@K, and Hit Rate@K for both systems, plus absolute metric deltas. It writes `reports/evaluation_summary.json` and `reports/evaluation_per_user.csv`.

Metrics are averaged across users with a relevant held-out item. The current protocol is a leave-one-last-interaction-out temporal test, not a full sequential ranking benchmark. Treat the results as a comparative offline evaluation, not proof of real-world user satisfaction.


## Local user feedback and personalization

The desktop app supports lightweight, local personalization:

- **Like** and **Dislike** buttons record explicit movie feedback.
- **Add to watchlist** saves a movie for later; the watchlist can be opened from the main window and entries can be removed.
- Feedback is stored in `.data/feedback.json` on the current machine. It is not uploaded or shared.
- The latest action for a movie is its active state. A later Like can replace a Dislike, and removing a watchlist entry does not erase its other history.
- Disliked and watchlisted movies are filtered from recommendations. Liked and disliked movie embeddings create a small preference profile that reranks candidates while retaining the original semantic/hybrid score.
- Use the **Local profile** field to keep feedback separate between people using the same installation. The MovieLens user ID remains a separate optional input for collaborative filtering.

The feedback layer is deliberately lightweight and does not retrain the transformer model. To reset a local profile, close the app and remove that profile's entry from `.data/feedback.json`.


## Dynamic user taste profile

The local recommender derives a dynamic metadata profile from each local profile's latest Like and Dislike states:

- Genre, cast, director, and keyword preferences are accumulated separately.
- Positive feedback increases affinity; negative feedback decreases it.
- Older feedback gradually loses influence (a 120-day half-life), allowing the profile to adapt as interests change.
- Watchlist entries are not treated as positive ratings; they remain excluded from recommendation results until removed.
- A **Taste profile** window summarizes learned likes and dislikes. Candidate ranking blends the existing semantic/hybrid score with metadata affinity.
- The profile is rebuilt from the local feedback file when needed; no model retraining or additional large model is required.

The profile is an interpretable preference summary based only on explicit movie feedback.


## Explainable recommendations

Each recommendation card now includes a **Why this movie?** section. Reasons are generated from signals actually available to the ranking pipeline:

- Semantic similarity to the current natural-language query.
- Matching genres, cast/directors, and story keywords from the user's explicit taste profile.
- Embedding similarity to a previously liked movie, when such feedback exists.
- Collaborative-filtering score, when the optional MovieLens model has a prediction for the candidate.

Explanations are deterministic and evidence-based; they do not use an LLM or invent reasons. A signal is omitted when the required evidence is unavailable. Similarity and collaborative values are ranking signals, not probabilities of liking a movie.


## Diversity-aware ranking (MMR)

After relevance and personalization scoring, the desktop app applies Maximal Marginal Relevance (MMR) to select a diverse top-k list. At each step, MMR balances a candidate's relevance with its maximum cosine similarity to movies already selected:

MMR = (1 - diversity_weight) * relevance - diversity_weight * redundancy

The GUI currently uses diversity_weight=0.35. Set it to 0 for relevance-only ranking; larger values place more emphasis on diversity. Relevance is taken from the best available score in this order: taste-profile, personalized feedback, hybrid, then semantic similarity. Embedding vectors are reused from the existing cache, so no additional model or embedding pass is needed.

MMR improves within-list diversity; it does not by itself establish that recommendations are novel or better-liked. Those outcomes should be measured in offline evaluation and user feedback.
