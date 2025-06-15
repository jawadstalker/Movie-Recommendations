import tkinter as tk
from PIL import Image, ImageTk
import requests
import io
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, left_on='id', right_on='movie_id')

descriptions = (movies['overview'].fillna('') + ' ' + 
                 movies['genres'].fillna('') + ' ' + 
                 movies['cast'].fillna('')).tolist()

embeddings = model.encode(descriptions)

def recommend_movies(text, topk=5):
    q_vec = model.encode([text])
    sims = cosine_similarity(q_vec, embeddings)[0]
    idx = sims.argsort()[::-1][:topk]
    return movies.iloc[idx]

def show_movies(text):
    recs = recommend_movies(text)
    window = tk.Toplevel()
    window.title("Movie Recommendations")
    window.geometry("800x1000")
    canvas = tk.Canvas(window, scrollregion=(0, 0, 800, 2000), bg="#edf2f4")
    canvas.pack(fill='both', expand=True)

    y = 20
    for i, row in recs.iterrows():
        poster_url = f"https://image.tmdb.org/t/p/w500/{row['poster_path']}"

        try:
            response = requests.get(poster_url)
            img = Image.open(io.BytesIO(response.content))
            img = img.resize((100, 150))
            photo = ImageTk.PhotoImage(img)
        except:
            photo = None

        if photo:
            lbl_img = tk.Label(canvas, image=photo, bd=2, relief='solid')
            lbl_img.image = photo
            lbl_img.place(x=20, y=y)

        lbl_title = tk.Label(canvas, text=row['title'], font=('Helvetica', 16, 'bold'), wraplength=500, justify='left'), 
        lbl_title = tk.Label(canvas, text=row['title'], font=('Helvetica', 16, 'bold'), wraplength=500, justify='left')
        lbl_title.place(x=150, y=y)

        lbl_overview = tk.Label(canvas, text=row['overview'], wraplength=500, justify='left', font=('Helvetica', 12))
        lbl_overview.place(x=150, y=y + 40)

        y += 200

def main():
    root = tk.Tk()
    root.title("Movie Recommender")
    root.geometry("500x250")
    root.config(bg="#edf2f4")

    lbl = tk.Label(root, text="Enter a sentence to find similar movies", font=('Helvetica', 14), bg="#edf2f4")
    lbl.pack(pady=20)

    entry = tk.Entry(root, width=50, font=('Helvetica', 14))
    entry.pack()

    button = tk.Button(root, text="Search", font=('Helvetica', 14), command=lambda: show_movies(entry.get()), bg="#ff9f1c")
    button.pack(pady=20)

    root.mainloop()

if __name__ == '__main__':
    main()
