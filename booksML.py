from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from waitress import serve  # Import the Waitress server

load_dotenv()
connection_string = os.getenv('MONGO_CONNECTION_STRING')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

client = MongoClient(connection_string)
db = client['media']
collection = db['books']

@app.route('/ping', methods=['GET'])
def ping():
    print("Pong")
    return jsonify({
        "msg": "Hello from the server"
    })

@app.route('/recommend', methods=['POST'])
def recommend_books():
    try:
        data = request.json
        app.logger.info("Received data: %s", data)

        book_id = data['book_id']

        all_books = list(collection.find())
        for book in all_books:
            book['_id'] = str(book['_id'])

        # Convert the list of books to a DataFrame
        df_books = pd.DataFrame(all_books)

        # Fill NaN values with an empty string to avoid JSON parsing errors
        df_books = df_books.fillna('')

        # Combine the title, authors, categories, and language into one feature (handling missing values)
        df_books['combined_features'] = df_books.apply(
            lambda row: ' '.join(filter(None, [row.get('title', ''), ' '.join(row.get('authors', [])), ' '.join(row.get('categories', [])), row.get('language', '')])),
            axis=1
        )

        # Create the TF-IDF vectorizer and transform the combined features
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_books['combined_features'])

        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Find the index of the input book
        try:
            input_book_index = df_books.index[df_books['_id'] == book_id][0]
        except IndexError:
            return jsonify({"error": "Book not found"}), 404

        # Get similarity scores for all books with the input book
        similarity_scores = list(enumerate(cosine_sim[input_book_index]))

        # Sort the books by similarity score (excluding the input book itself)
        similar_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Drop the 'combined_features' column before returning the final results
        df_books = df_books.drop(columns=['combined_features', "__v"])

        # Get the top 10 most similar books, excluding the input book itself, and return full book data
        top_similar_books = [df_books.iloc[i].to_dict() for i, score in similar_books if i != input_book_index][:10]

        # Return the result as a JSON-encoded list of book objects
        return jsonify(top_similar_books)
    except Exception as e:
        app.logger.error("Error processing request: %s", e)
        return jsonify({"error": str(e)}), 403

# Run the application using Waitress if this file is executed directly
if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8080)  # Use Waitress to serve the app
