from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the trained model and necessary data
model = joblib.load('saved_models/model.joblib')
data = pd.read_csv('saved_models/Amazon_BooksDataset.csv')  # Make sure this file is in the same directory
new_data = joblib.load('saved_models/new_data.joblib')  # We'll need to save this from your notebook
similarity_matrix = joblib.load('saved_models/similarity_matrix.joblib')  # We'll need to save this from your notebook

# Your helper functions
def full_title(user_input):
    book_titles = list(set(new_data['Book Name']))
    for books in book_titles:
        if user_input.lower() == books.lower():
            return books
    words = user_input.split()
    user_input = max(words, key=len)
    for title in book_titles:
        if user_input.lower() in title.lower():
            return title
    return "Not Found"

def get_recommendations(book_name, similarity_matrix, top_n=5):
    book_name = full_title(book_name)
    if book_name == 'Not Found':
        return ['No match found']
    book_idx = new_data[new_data['Book Name'] == book_name].index[0]
    similar_books = list(enumerate(similarity_matrix[book_idx]))
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
    recommended_books = [new_data['Book Name'][i] for i, score in similar_books[1:top_n+1]]
    return recommended_books

def truncate_title(title, max_words=12):
    words = title.split()
    if len(words) > max_words:
        truncated_title = ' '.join(words[:max_words]) + '...'
        return truncated_title
    else:
        return title

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    user_input = request.form['userInput']
    recommendations = get_recommendations(user_input, similarity_matrix)
    truncated_recommendations = [truncate_title(book) for book in recommendations]
    return jsonify({
        'recommendations': truncated_recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)