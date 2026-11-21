import argparse

from sentence_transformers import SentenceTransformer

from evaluation import evaluate_recommenders, save_evaluation_report
from hybrid import load_movielens_data
from recommender import (
    MODEL_NAME,
    build_movie_texts,
    get_movie_embeddings,
    load_movies,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare content-based and hybrid movie recommendation quality."
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--min-train-ratings", type=int, default=3)
    parser.add_argument("--content-weight", type=float, default=0.65)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    movies = load_movies()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_movie_embeddings(build_movie_texts(movies), model)
    ratings, links = load_movielens_data()

    results = evaluate_recommenders(
        movies=movies,
        embeddings=embeddings,
        model=model,
        ratings=ratings,
        links=links,
        k=args.k,
        relevance_threshold=args.threshold,
        min_train_ratings=args.min_train_ratings,
        content_weight=args.content_weight,
        max_users=args.max_users,
    )
    output = save_evaluation_report(results)

    print(f"Evaluation split: {results['split']}")
    print(f"Users evaluated: {results['users_evaluated']}")
    print(f"Ranking cutoff: K={results['k']}")
    print(f"Relevance threshold: {results['relevance_threshold']}")
    print()
    print(f"{'Metric':<14}{'Content-Based':>16}{'Hybrid':>14}{'Delta':>12}")
    for metric in ("precision", "recall", "map", "ndcg", "hit_rate"):
        before = results["content_based"][metric]
        after = results["hybrid"][metric]
        delta = results["improvement"][metric]
        print(f"{metric:<14}{before:>16.4f}{after:>14.4f}{delta:>+12.4f}")
    print()
    print(f"Reports saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
