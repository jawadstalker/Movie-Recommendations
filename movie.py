import io
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from urllib.parse import quote

import requests
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer

from recommender import MODEL_NAME, build_movie_texts, get_movie_embeddings, load_movies, recommend_movies

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
        self.poster_images = []

        ttk.Label(root, text="Describe the movie you want to watch", font=("Helvetica", 16, "bold")).pack(pady=(24, 12))
        self.query = tk.Entry(root, width=58, font=("Helvetica", 12))
        self.query.pack(ipady=7, padx=24)
        self.query.bind("<Return>", lambda _event: self.search())

        controls = ttk.Frame(root)
        controls.pack(pady=14)
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
            self.root.after(0, lambda: self.ready(movies, model, embeddings))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self.failed(error))

    def ready(self, movies, model, embeddings):
        self.movies, self.model, self.embeddings = movies, model, embeddings
        self.status.set(f"Ready — {len(movies):,} movies loaded. Embeddings are cached for the next launch.")
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
            results = recommend_movies(query, self.movies, self.embeddings, self.model, top_k=self.count.get())
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
            score = float(movie.get("similarity", 0))
            ttk.Label(details, text=f"{rank}. {title} ({year})", font=("Helvetica", 14, "bold")).pack(anchor="w")
            ttk.Label(details, text=f"Semantic match: {score:.1%}", foreground="#555").pack(anchor="w", pady=(3, 6))
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
