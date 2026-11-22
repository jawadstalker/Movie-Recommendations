import io
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from urllib.parse import quote

import requests
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer

from recommender import MODEL_NAME, build_movie_texts, get_movie_embeddings, load_movies
from hybrid import load_movielens_data, build_collaborative_model, recommend_hybrid
from feedback import (
    apply_feedback,
    get_user_feedback,
    get_watchlist,
    record_feedback,
)
from taste_profile import build_taste_profile, apply_taste_profile, describe_taste_profile
from explainability import explain_recommendations
from mmr import select_diverse_recommendations
from ab_testing import assign_variant, record_exposure, record_outcome, summarize_experiment

BG = "#edf2f4"
ACCENT = "#ff9f1c"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"


class MovieApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommender")
        self.root.geometry("680x330")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.model = None
        self.movies = None
        self.embeddings = None
        self.collaborative_model = None
        self.poster_images = []
        self.profile_id = tk.StringVar(value="default")
        self.current_impressions = {}

        ttk.Label(root, text="Describe the movie you want to watch", font=("Helvetica", 16, "bold")).pack(pady=(20, 10))
        self.query = tk.Entry(root, width=58, font=("Helvetica", 12))
        self.query.pack(ipady=7, padx=24)
        self.query.bind("<Return>", lambda _event: self.search())

        user_controls = ttk.Frame(root)
        user_controls.pack(pady=(9, 0))
        ttk.Label(user_controls, text="MovieLens user ID (optional)").pack(side="left", padx=(0, 8))
        self.user_id = tk.StringVar()
        ttk.Entry(user_controls, textvariable=self.user_id, width=12).pack(side="left", padx=(0, 18))
        ttk.Label(user_controls, text="Local profile").pack(side="left", padx=(0, 8))
        ttk.Entry(user_controls, textvariable=self.profile_id, width=14).pack(side="left")

        controls = ttk.Frame(root)
        controls.pack(pady=10)
        ttk.Label(controls, text="Results").pack(side="left", padx=(0, 8))
        self.count = tk.IntVar(value=5)
        ttk.Spinbox(controls, from_=1, to=20, width=5, textvariable=self.count).pack(side="left")
        self.search_button = tk.Button(controls, text="Find movies", command=self.search, bg=ACCENT, relief="flat", padx=18, pady=8, state="disabled")
        self.search_button.pack(side="left", padx=12)
        tk.Button(controls, text="My watchlist", command=self.show_watchlist, relief="flat", padx=12, pady=8).pack(side="left", padx=(0, 6))
        tk.Button(controls, text="Taste profile", command=self.show_taste_profile, relief="flat", padx=12, pady=8).pack(side="left")
        tk.Button(controls, text="A/B report", command=self.show_experiment_report, relief="flat", padx=12, pady=8).pack(side="left", padx=(6, 0))
        self.status = tk.StringVar(value="Loading movie data and language model…")
        ttk.Label(root, textvariable=self.status, wraplength=620).pack(pady=8)
        threading.Thread(target=self.initialize, daemon=True).start()

    def initialize(self):
        try:
            movies = load_movies()
            model = SentenceTransformer(MODEL_NAME)
            texts = build_movie_texts(movies)
            embeddings = get_movie_embeddings(texts, model)
            collaborative = None
            from pathlib import Path
            if Path("ratings.csv").is_file() and Path("links.csv").is_file():
                try:
                    ratings, links = load_movielens_data()
                    collaborative = build_collaborative_model(ratings, links, movies)
                except Exception as exc:
                    print(f"Collaborative model unavailable: {exc}")
            self.root.after(0, lambda: self.ready(movies, model, embeddings, collaborative))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self.failed(error))

    def ready(self, movies, model, embeddings, collaborative=None):
        self.movies, self.model, self.embeddings = movies, model, embeddings
        self.collaborative_model = collaborative
        mode = "Hybrid recommendations available." if collaborative else "Content-based mode (add MovieLens ratings.csv and links.csv for hybrid recommendations)."
        self.status.set(f"Ready — {len(movies):,} movies loaded. {mode}")
        self.search_button.configure(state="normal")

    def failed(self, error):
        self.status.set("Initialization failed.")
        self.search_button.configure(state="disabled")
        messagebox.showerror("Could not start Movie Recommender", error)

    def _active_profile(self):
        return self.profile_id.get().strip() or "default"

    def search(self):
        query = self.query.get().strip()
        if not query:
            messagebox.showinfo("Movie Recommender", "Describe the kind of movie you want to watch.")
            return
        if self.model is None:
            messagebox.showinfo("Please wait", "The movie data and model are still loading.")
            return
        try:
            raw_user_id = self.user_id.get().strip()
            user_id = int(raw_user_id) if raw_user_id.isdigit() else raw_user_id or None
            results = recommend_hybrid(
                query, self.movies, self.embeddings, self.model,
                user_id=user_id, collaborative_model=self.collaborative_model,
                top_k=len(self.movies),
            )
            profile_id = self._active_profile()
            variant = assign_variant(profile_id)
            feedback = get_user_feedback(profile_id)
            results = apply_feedback(results, self.embeddings, feedback, top_k=len(results))
            profile = build_taste_profile(self.movies, feedback)
            results = apply_taste_profile(results, profile, top_k=len(results))
            if variant == "B":
                results = select_diverse_recommendations(
                    results, self.embeddings, top_k=self.count.get(),
                    diversity_weight=0.35,
                )
            else:
                results = results.head(self.count.get()).copy()
            results = explain_recommendations(
                results, self.movies, self.embeddings,
                feedback=feedback, taste_profile=profile,
            )
            self.current_impressions = {}
            if not results.empty:
                try:
                    exposure = record_exposure(profile_id, variant, results["id"].tolist())
                    self.current_impressions = {
                        int(movie_id): exposure["impression_id"]
                        for movie_id in results["id"].tolist()
                    }
                except Exception as logging_error:
                    print(f"A/B exposure logging unavailable: {logging_error}")
            self.status.set(
                f"Ready — A/B experiment {variant}: "
                f"{'baseline ranking' if variant == 'A' else 'MMR diversity ranking'}."
            )
            self.show_results(results)
        except Exception as exc:
            messagebox.showerror("Recommendation error", str(exc))

    def save_feedback(self, movie, action, status, impression_id=None):
        try:
            movie_id = int(movie["id"])
            title = str(movie.get("title", "Untitled"))
            profile_id = self._active_profile()
            record_feedback(profile_id, movie_id, title, action)
            if impression_id:
                try:
                    record_outcome(profile_id, impression_id, movie_id, action)
                except Exception as logging_error:
                    print(f"A/B outcome logging unavailable: {logging_error}")
            status.set({
                "like": "Liked — future recommendations will adapt.",
                "dislike": "Disliked — this movie will be hidden.",
                "watchlist": "Saved to your watchlist.",
            }[action])
        except Exception as exc:
            messagebox.showerror("Could not save feedback", str(exc))

    def show_results(self, results):
        window = tk.Toplevel(self.root)
        window.title("Your movie recommendations")
        window.geometry("920x780")
        canvas = tk.Canvas(window, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.poster_images = []

        for rank, (movie_index, movie) in enumerate(results.iterrows(), start=1):
            card = ttk.Frame(content, padding=14)
            card.pack(fill="x", padx=12, pady=8)
            poster = self.load_poster(movie.get("poster_path"))
            if poster:
                self.poster_images.append(poster)
                ttk.Label(card, image=poster).pack(side="left", padx=(0, 16), anchor="n")
            details = ttk.Frame(card)
            details.pack(side="left", fill="x", expand=True)
            title = str(movie.get("title", "Untitled"))
            year = str(movie.get("release_date", ""))[:4]
            score = float(movie.get("personalized_score", movie.get("hybrid_score", movie.get("similarity", 0))))
            method = str(movie.get("recommendation_type", "content-based"))
            ttk.Label(details, text=f"{rank}. {title} ({year})", font=("Helvetica", 14, "bold")).pack(anchor="w")
            ttk.Label(details, text=f"Recommendation score: {score:.1%} · {method}", foreground="#555").pack(anchor="w", pady=(3, 6))
            overview = str(movie.get("overview", "") or "No overview available.")
            ttk.Label(details, text=overview, wraplength=650, justify="left").pack(anchor="w")
            reasons = movie.get("explanations", [])
            if isinstance(reasons, list) and reasons:
                explanation_box = ttk.LabelFrame(details, text="Why this movie?", padding=8)
                explanation_box.pack(fill="x", pady=(7, 0))
                for reason in reasons:
                    ttk.Label(
                        explanation_box, text="• " + str(reason),
                        wraplength=630, justify="left",
                    ).pack(anchor="w", pady=1)
            feedback_status = tk.StringVar(value="")
            actions = ttk.Frame(details)
            actions.pack(anchor="w", pady=(8, 0))
            impression_id = self.current_impressions.get(int(movie["id"]))
            tk.Button(actions, text="Like", command=lambda m=movie.copy(), s=feedback_status, i=impression_id: self.save_feedback(m, "like", s, i)).pack(side="left", padx=(0, 5))
            tk.Button(actions, text="Dislike", command=lambda m=movie.copy(), s=feedback_status, i=impression_id: self.save_feedback(m, "dislike", s, i)).pack(side="left", padx=5)
            tk.Button(actions, text="Add to watchlist", command=lambda m=movie.copy(), s=feedback_status, i=impression_id: self.save_feedback(m, "watchlist", s, i)).pack(side="left", padx=5)
            ttk.Label(details, textvariable=feedback_status, foreground="#287a48").pack(anchor="w", pady=(4, 0))

    def show_experiment_report(self):
        try:
            report = summarize_experiment()
            variants = report["variants"]
            lines = [
                "A/B experiment: " + report["experiment"],
                "",
                "Rates are based on recommended movie impressions, not unique visitors.",
                "",
            ]
            for variant, label in (("A", "Baseline (without MMR)"), ("B", "MMR (diversity weight 0.35)")):
                data = variants[variant]
                lines.extend([
                    f"{variant} — {label}",
                    f"  Impressions: {data['impressions']}",
                    f"  Profiles: {data['profiles']}",
                    f"  Recommended items: {data['recommended_items']}",
                    f"  Likes: {data['likes']} ({data['like_rate']:.1%})",
                    f"  Dislikes: {data['dislikes']} ({data['dislike_rate']:.1%})",
                    f"  Watchlist adds: {data['watchlist_adds']} ({data['watchlist_rate']:.1%})",
                    f"  Explicit engagement rate: {data['engagement_rate']:.1%}",
                    "",
                ])
            lines.append("Treat small samples as directional; this report does not calculate statistical significance.")
            messagebox.showinfo("A/B experiment report", "\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Could not build A/B report", str(exc))

    def show_taste_profile(self):
        try:
            feedback = get_user_feedback(self._active_profile())
            profile = build_taste_profile(self.movies, feedback)
            window = tk.Toplevel(self.root)
            window.title("My taste profile")
            window.geometry("620x620")
            ttk.Label(window, text=f"Taste profile · {self._active_profile()}",
                      font=("Helvetica", 15, "bold")).pack(pady=(16, 6))
            ttk.Label(window, text=f"{profile['liked_count']} liked · {profile['disliked_count']} disliked",
                      foreground="#555").pack(pady=(0, 12))
            if profile["feedback_count"] == 0:
                ttk.Label(window, text="Your profile is still empty. Like or dislike a few recommendations to help the system learn.",
                          wraplength=540, justify="center").pack(padx=24, pady=24)
                return
            for label, positive, negative in describe_taste_profile(profile):
                section = ttk.LabelFrame(window, text=label, padding=12)
                section.pack(fill="x", padx=18, pady=8)
                positive_text = ", ".join(name for name, _ in positive) if positive else "Not enough evidence yet"
                negative_text = ", ".join(name for name, _ in negative) if negative else "None identified"
                ttk.Label(section, text="Likes: " + positive_text, wraplength=520).pack(anchor="w")
                ttk.Label(section, text="Avoids: " + negative_text, wraplength=520).pack(anchor="w", pady=(5, 0))
        except Exception as exc:
            messagebox.showerror("Could not build taste profile", str(exc))

    def show_watchlist(self):
        try:
            items = get_watchlist(get_user_feedback(self._active_profile()))
            window = tk.Toplevel(self.root)
            window.title("My watchlist")
            window.geometry("520x520")
            ttk.Label(window, text=f"Watchlist · {self._active_profile()}", font=("Helvetica", 14, "bold")).pack(pady=12)
            if not items:
                ttk.Label(window, text="Your watchlist is empty.").pack(pady=20)
                return
            for item in items:
                row = ttk.Frame(window, padding=8)
                row.pack(fill="x", padx=12)
                ttk.Label(row, text=item.get("title", f"Movie {item['movie_id']}")).pack(side="left", fill="x", expand=True)
                tk.Button(
                    row, text="Remove",
                    command=lambda movie=item, parent=row: self.remove_from_watchlist(movie, parent),
                ).pack(side="right")

        except Exception as exc:
            messagebox.showerror("Could not load watchlist", str(exc))

    def remove_from_watchlist(self, item, row):
        try:
            record_feedback(self._active_profile(), item["movie_id"], item.get("title", ""), "remove_watchlist")
            row.destroy()
        except Exception as exc:
            messagebox.showerror("Could not update watchlist", str(exc))

    @staticmethod
    def load_poster(path):
        if not isinstance(path, str) or not path.strip() or path == "nan":
            return None
        try:
            response = requests.get(f"{POSTER_BASE}/{quote(path.lstrip('/'))}", timeout=8)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            image.thumbnail((120, 180))
            return ImageTk.PhotoImage(image)
        except (requests.RequestException, OSError, ValueError):
            return None


def main():
    root = tk.Tk()
    MovieApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
