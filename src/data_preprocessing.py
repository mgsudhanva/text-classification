import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess a given text as per the training preprocessing."""
    # Normalize: convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenization and removing stopwords
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Reconstruct the text
    return ' '.join(filtered_tokens)

# Load the trained TF-IDF Vectorizer
tfidf_vectorizer = joblib.load('../models/vectorizer.joblib')

def vectorize_text(text):
    """Convert text to its TF-IDF vector representation."""
    return tfidf_vectorizer.transform([text])
