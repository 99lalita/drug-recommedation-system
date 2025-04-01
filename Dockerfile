# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# ✅ Install SQLite & net-tools (for netstat debugging)
RUN apt-get update && apt-get install -y sqlite3 net-tools

# ✅ Copy project files into the container
COPY . /app

# ✅ Copy trained models (Fix: Include TF-IDF files)
COPY models/sentiment_model.pkl models/vectorizer.pkl models/tfidf_vectorizer.pkl models/tfidf_matrix.pkl /app/models/

# ✅ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Expose Flask port
EXPOSE 5000

# ✅ Start the application (Fix: Run Flask instead of Gunicorn)
CMD ["python", "app.py"]
