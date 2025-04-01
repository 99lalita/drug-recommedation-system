# Import libraries
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load datasets
try:
    logging.info("✅ Loading dataset...")
    df_train = pd.read_csv('dataset/drugsComTrain_raw.csv', encoding="utf-8", engine="python", on_bad_lines="skip")
    df_test = pd.read_csv('dataset/drugsComTest_raw.csv', encoding="utf-8", engine="python", on_bad_lines="skip")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df[['drugName', 'condition', 'review', 'rating']]
    df.rename(columns={'drugName': 'drug_name'}, inplace=True)
    df.dropna(inplace=True)

    logging.info(f"✅ Dataset loaded successfully! {df.shape[0]} records found.")

except Exception as e:
    logging.error(f"❌ Error loading dataset: {e}")
    df = pd.DataFrame(columns=['drug_name', 'condition', 'review', 'rating'])

# ✅ Define text preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(clean_text)

# ✅ Remove short reviews (less than 5 words)
df = df[df['clean_review'].apply(lambda x: len(x.split()) >= 5)]

# ✅ Label sentiment (Positive, Neutral, Negative)
def get_sentiment_label(rating):
    if rating >= 7:
        return 'positive'
    elif rating >= 4:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating'].apply(get_sentiment_label)

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['sentiment'], test_size=0.2, random_state=42)

# ✅ Feature Extraction using Optimized TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))  # ✅ Reduced from 5000 to 3000
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ✅ Apply SMOTE (Only for class imbalance correction)
smote = SMOTE(sampling_strategy='auto', random_state=42)  # ✅ Prevents over-sampling of already balanced classes
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

# ✅ Train a RandomForestClassifier with Optimized Bayesian Search
def rf_evaluate(n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        n_jobs=-1,  # ✅ Uses all CPU cores for faster training
        random_state=42
    )
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_tfidf)
    return f1_score(y_test, y_pred, average='weighted')

# ✅ Bayesian Optimization for best hyperparameters (Reduced Iterations)
optimizer = BayesianOptimization(
    f=rf_evaluate,
    pbounds={'n_estimators': (50, 200), 'max_depth': (5, 30)},  # ✅ Reduced range to speed up search
    random_state=42,
)
optimizer.maximize(init_points=3, n_iter=5)  # ✅ Reduced from `10` to `5` for faster training

# ✅ Get best parameters
best_params = optimizer.max['params']
best_n_estimators = int(best_params['n_estimators'])
best_max_depth = int(best_params['max_depth'])

# ✅ Train final model with best parameters
final_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, n_jobs=-1, random_state=42)
final_model.fit(X_train_balanced, y_train_balanced)

# ✅ Save trained model and vectorizer
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ✅ Evaluate model
y_pred = final_model.predict(X_test_tfidf)

logging.info("\n✅ **Model Performance Metrics:**")
logging.info(f"🔹 Accuracy: {accuracy_score(y_test, y_pred)}")
logging.info(f"🔹 Precision: {precision_score(y_test, y_pred, average='weighted')}")
logging.info(f"🔹 Recall: {recall_score(y_test, y_pred, average='weighted')}")
logging.info(f"🔹 F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

logging.info("🚀 Sentiment model training complete!")
