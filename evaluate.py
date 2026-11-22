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
    parser.add_argument(
        "--mmr-weights", type=str, default="0.20,0.35,0.50",
        help="Comma-separated diversity weights to compare, from 0 to 1.",
    )
    parser.add_argument(
        "--mmr-candidate-pool", type=int, default=100,
        help="Number of top hybrid candidates considered by MMR per user.",
    )
    args = parser.parse_args()
    try:
        mmr_weights = tuple(float(value.strip()) for value in args.mmr_weights.split(",") if value.strip())
    except ValueError as exc:
        parser.error(f"Invalid --mmr-weights value: {exc}")

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
        mmr_weights=mmr_weights,
        mmr_candidate_pool=args.mmr_candidate_pool,
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
    print("MMR parameter sweep (hybrid relevance shortlist):")
    print(f"{'Weight':<10}{'NDCG@K':>12}{'MAP@K':>12}{'Precision@K':>16}{'ILD':>10}")
    for weight, values in results["mmr"].items():
        metrics = values["ranking"]
        print(
            f"{weight:<10}{metrics['ndcg']:>12.4f}{metrics['map']:>12.4f}"
            f"{metrics['precision']:>16.4f}{values['intra_list_diversity']:>10.4f}"
        )
    print()
    print(f"Reports saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
