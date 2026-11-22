from datetime import datetime, timezone
import pandas as pd
from taste_profile import build_taste_profile, apply_taste_profile, describe_taste_profile

def sample_movies():
    return pd.DataFrame([
        {"id":1,"title":"Space Drama","genres":'[{"name":"Drama"},{"name":"Science Fiction"}]',
         "cast":'[{"name":"Actor A"},{"name":"Actor B"}]',
         "keywords":'[{"name":"space travel"},{"name":"identity"}]',
         "crew":'[{"name":"Director A","job":"Director"}]'},
        {"id":2,"title":"Space Comedy","genres":'[{"name":"Comedy"},{"name":"Science Fiction"}]',
         "cast":'[{"name":"Actor A"}]',"keywords":'[{"name":"space travel"},{"name":"friendship"}]',
         "crew":'[{"name":"Director B","job":"Director"}]'},
        {"id":3,"title":"Crime Story","genres":'[{"name":"Crime"},{"name":"Thriller"}]',
         "cast":'[{"name":"Actor C"}]',"keywords":'[{"name":"detective"}]',
         "crew":'[{"name":"Director C","job":"Director"}]'},
    ])

def test_profile_learns_metadata_preferences():
    movies = sample_movies()
    now = datetime(2026,11,22,tzinfo=timezone.utc)
    profile = build_taste_profile(movies, [
        {"movie_id":1,"action":"like","timestamp":now.isoformat()},
        {"movie_id":3,"action":"dislike","timestamp":now.isoformat()}], now=now)
    assert profile["liked_count"] == 1 and profile["disliked_count"] == 1
    assert profile["preferences"]["genres"]["drama"] > 0
    assert profile["preferences"]["genres"]["crime"] < 0
    assert profile["preferences"]["people"]["director a"] > 0
    assert profile["preferences"]["keywords"]["space travel"] > 0

def test_profile_reranks_using_learned_preferences():
    movies = sample_movies()
    now = datetime(2026,11,22,tzinfo=timezone.utc)
    profile = build_taste_profile(movies,[{"movie_id":1,"action":"like","timestamp":now.isoformat()}],now=now)
    candidates = movies.iloc[[1,2]].copy()
    candidates["similarity"] = [0.8,0.8]
    ranked = apply_taste_profile(candidates,profile,top_k=2,profile_weight=0.5)
    assert ranked.iloc[0]["id"] == 2
    assert "taste_profile_score" in ranked.columns
    assert len(describe_taste_profile(profile)) == 3

def test_watchlist_is_not_a_positive_preference():
    profile = build_taste_profile(sample_movies(),[{"movie_id":1,"action":"watchlist"}])
    assert profile["feedback_count"] == 0
