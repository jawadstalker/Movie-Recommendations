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

BG = "#edf2f4"
ACCENT = "#ff9f1c"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"


class MovieApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommender")
        self.root.geometry("650x280")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.model = None
        self.movies = None
        self.embeddings = None
        self.collaborative_model = None
        self.poster_images = []

        ttk.Label(root, text="Describe the movie you want to watch", font=("Helvetica", 16, "bold")).pack(pady=(24, 12))
        self.query = tk.Entry(root, width=58, font=("Helvetica", 12))
        self.query.pack(ipady=7, padx=24)
        self.query.bind("<Return>", lambda _event: self.search())

        user_controls = ttk.Frame(root)
        user_controls.pack(pady=(10, 0))
        ttk.Label(user_controls, text="MovieLens user ID (optional)").pack(side="left", padx=(0, 8))
        self.user_id = tk.StringVar()
        ttk.Entry(user_controls, textvariable=self.user_id, width=14).pack(side="left")

        controls = ttk.Frame(root)
        controls.pack(pady=10)
        ttk.Label(controls, text="Results").pack(side="left", padx=(0, 8))
        self.count = tk.IntVar(value=5)
        ttk.Spinbox(controls, from_=1, to=20, width=5, textvariable=self.count).pack(side="left")
        self.search_button = tk.Button(controls, text="Find movies", command=self.search, bg=ACCENT, relief="flat", padx=18, pady=8)
        self.search_button.pack(side="left", padx=16)
        self.status = tk.StringVar(value="Loading movie data and language model…")
        ttk.Label(root, textvariable=self.status, wraplength=580).pack(pady=8)
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
                except Exception:
                    collaborative = None
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
                top_k=self.count.get(),
            )
            self.show_results(results)
        except Exception as exc:
            messagebox.showerror("Recommendation error", str(exc))

    def show_results(self, results):
        window = tk.Toplevel(self.root)
        window.title("Your movie recommendations")
        window.geometry("900x760")
        canvas = tk.Canvas(window, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.poster_images = []

        for rank, (_, movie) in enumerate(results.iterrows(), start=1):
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
            score = float(movie.get("hybrid_score", movie.get("similarity", 0)))
            method = str(movie.get("recommendation_type", "content-based"))
            ttk.Label(details, text=f"{rank}. {title} ({year})", font=("Helvetica", 14, "bold")).pack(anchor="w")
            ttk.Label(details, text=f"Recommendation score: {score:.1%} · {method}", foreground="#555").pack(anchor="w", pady=(3, 6))
            overview = str(movie.get("overview", "") or "No overview available.")
            ttk.Label(details, text=overview, wraplength=650, justify="left").pack(anchor="w")

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
