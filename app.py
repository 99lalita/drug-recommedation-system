from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib  # ✅ Faster than pickle
import logging
import sqlite3
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# ✅ Optimize Dataset Loading
try:
    logging.info("✅ Loading dataset...")

    dtypes = {
        'drugName': 'string',
        'condition': 'string',
        'review': 'string',
        'rating': 'float32'
    }

    df_train = pd.read_csv('dataset/drugsComTrain_raw.csv', encoding="utf-8", dtype=dtypes, usecols=dtypes.keys())
    df_test = pd.read_csv('dataset/drugsComTest_raw.csv', encoding="utf-8", dtype=dtypes, usecols=dtypes.keys())

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.rename(columns={'drugName': 'drug_name'}, inplace=True)
    df.dropna(inplace=True)

    logging.info(f"✅ Dataset loaded! {df.shape[0]} records found.")

except Exception as e:
    logging.error(f"❌ Error loading dataset: {e}")
    df = pd.DataFrame(columns=['drug_name', 'condition', 'review', 'rating'])

# ✅ Optimize Model Loading
try:
    logging.info("✅ Loading models...")

    sentiment_model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

    logging.info("✅ Models loaded successfully!")

except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    sentiment_model, vectorizer, tfidf_vectorizer, tfidf_matrix = None, None, None, None

# ✅ Function to predict sentiment
def get_sentiment(text):
    try:
        if sentiment_model and vectorizer:
            text_vectorized = vectorizer.transform([text])
            return sentiment_model.predict(text_vectorized)[0]
    except Exception as e:
        logging.error(f"❌ Error in sentiment analysis: {e}")
    
    return "neutral"  # ✅ Prevents NoneType errors

df['sentiment'] = df['review'].apply(get_sentiment)

# ✅ Load TF-IDF Matrix (with debugging)
try:
    logging.info("✅ Loading TF-IDF matrix...")
    tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
except Exception as e:
    logging.error(f"❌ Error loading TF-IDF matrix: {e}")
    tfidf_matrix = None

# ✅ Database Initialization
def initialize_database():
    try:
        with sqlite3.connect('database.db', timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS drugs (
                                id INTEGER PRIMARY KEY,
                                drug_name TEXT,
                                condition TEXT,
                                review TEXT,
                                rating REAL,
                                sentiment TEXT)''')
            conn.commit()
        logging.info("✅ Database initialized successfully!")

    except Exception as e:
        logging.error(f"❌ Database error: {e}")

# ✅ Function to insert data into the database
def save_to_db():
    try:
        with sqlite3.connect('database.db', timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM drugs")  # ✅ Prevent duplicate data
            for _, row in df.iterrows():
                cursor.execute('''INSERT INTO drugs (drug_name, condition, review, rating, sentiment)
                                  VALUES (?, ?, ?, ?, ?)''',
                               (row['drug_name'], row['condition'], row['review'], row['rating'], row['sentiment']))
            conn.commit()
        logging.info("✅ Data inserted into database!")

    except Exception as e:
        logging.error(f"❌ Error inserting into database: {e}")

initialize_database()
save_to_db()

# ✅ Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if not tfidf_vectorizer or not tfidf_matrix:
            return jsonify({"error": "TF-IDF model not loaded"}), 500

        user_input = request.form['user_input']
        user_sentiment_label = get_sentiment(user_input)

        if not user_sentiment_label:
            return jsonify({"error": "Unable to determine sentiment"}), 400

        # ✅ Convert user input into a vector
        user_vector = tfidf_vectorizer.transform([user_input])

        # ✅ Compute similarity scores
        similarity_scores = cosine_similarity(user_vector, tfidf_matrix)

        # ✅ Get top 5 most similar drugs
        top_indices = similarity_scores.argsort()[0][-5:][::-1]
        recommendations = df.iloc[top_indices]

        # ✅ Filter recommendations by sentiment
        recommendations = recommendations[recommendations['sentiment'] == user_sentiment_label]

        if recommendations.empty:
            return jsonify({"message": "No recommendations found matching your sentiment"}), 404

        recommended_drugs = recommendations[['drug_name', 'condition', 'rating', 'sentiment']].to_dict(orient='records')

        return jsonify(recommended_drugs)

    except Exception as e:
        logging.error(f"❌ Error in recommendation: {e}")
        return jsonify({"error": str(e)}), 500

logging.info("✅ Running Flask on http://0.0.0.0:5000 🚀")

if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
    except Exception as e:
        logging.error(f"❌ Critical Error: {e}")
