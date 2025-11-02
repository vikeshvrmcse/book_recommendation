import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load datasets
popular_df = pickle.load(open('./models/popular.pkl', 'rb'))
books = pickle.load(open('./models/books.pkl', 'rb'))
pt = pickle.load(open('./models/pt.pkl', 'rb'))
similarity_score = pickle.load(open('./models/similarity_score.pkl', 'rb'))

app = Flask(__name__)

# --- Helper Functions ---
def is_internet_connected():
    response = os.system("ping -c 1 google.com" if os.name != "nt" else "ping -n 1 google.com")
    return response == 0

def get_similar_books(book_name):
    """Return similar books as a list of dictionaries with additional features"""
    indices = np.where(pt.index == book_name)[0]
    if len(indices) == 0:
        return None
    index = indices[0]
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        if not temp_df.empty:
            item = {
                "title": temp_df['Book-Title'].values[0],
                "author": temp_df['Book-Author'].values[0],
                "image": temp_df['Image-URL-M'].values[0],
                "publisher": temp_df['Publisher'].values[0] if 'Publisher' in temp_df.columns else "N/A",
                "language": temp_df['Language'].values[0] if 'Language' in temp_df.columns else "N/A",
                "ratings": temp_df['Ratings'].values[0] if 'Ratings' in temp_df.columns else "N/A",
                "download_link": temp_df['Download-URL'].values[0] if 'Download-URL' in temp_df.columns else "N/A",
                "buy_link": temp_df['Buy-URL'].values[0] if 'Buy-URL' in temp_df.columns else "N/A"
            }
            data.append(item)
    return data

# --- Routes ---
@app.route('/')
def home():
    return "Welcome, Tata By By!"

@app.route('/BookRecommender', methods=["POST"])
def book_recommender():
    data = request.get_json()
    entered_book_name = data.get("book") if data else None

    if not entered_book_name or len(entered_book_name.strip()) == 0:
        return jsonify({"error": "Please enter the book name and try again."}), 400

    recommendations = get_similar_books(entered_book_name)
    if recommendations:
        return jsonify({
            "query": entered_book_name,
            "recommendations": recommendations
        })
    else:
        return jsonify({"error": "Book not found. Please check the book name and try again."}), 404

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
