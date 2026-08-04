import sys
import os

# Add project root directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.scrapers.fetcher import fetch_website_content

def train_ml_model():
    print("Initializing model training pipeline...")
    dataset_path = "data/FR_online_market_discovery_2024.parquet"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Using fallback corpus.")
        corpus = [
            "buy products add to cart free shipping secure checkout price product item boutique livraison commander",
            "shop online discount store items add to basket sale add to cart catalog buy now cart",
            "latest news articles blog post daily update editorial contact us privacy policy terms search",
            "company profile global services team about us leadership career investors corporate details"
        ]
        labels = [1, 1, 0, 0]
    else:
        df = pd.read_parquet(dataset_path)
        urls = df['url'].dropna().unique()[:30]  # Sample 30 domains for training data
        
        corpus = []
        labels = []
        
        strong_keywords = {
            'add to cart', 'add to basket', 'checkout', 'shopping cart', 'view cart', 
            'buy now', 'my cart', 'panier', 'commander', 'ajouter au panier', 
            'achat en ligne', 'mon panier', 'free shipping', 'livraison gratuite'
        }
        
        print(f"Bootstrapping training data from {len(urls)} live domains...")
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            data = fetch_website_content(url)
            if data and len(data["raw_text"]) > 100:
                text = data["raw_text"]
                is_shop = any(kw in text for kw in strong_keywords)
                corpus.append(text)
                labels.append(1 if is_shop else 0)

        if len(corpus) < 5:
            corpus = [
                "buy products add to cart free shipping secure checkout price product item boutique livraison commander",
                "shop online discount store items add to basket sale add to cart catalog buy now cart",
                "latest news articles blog post daily update editorial contact us privacy policy terms search",
                "company profile global services team about us leadership career investors corporate details"
            ]
            labels = [1, 1, 0, 0]

    # Vectorize and train
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus)
    
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X, labels)

    # Save serialized model files
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(vectorizer, "src/models/tfidf_vectorizer.pkl")
    joblib.dump(clf, "src/models/logistic_model.pkl")
    print("Successfully trained and saved model artifacts to src/models/!")

if __name__ == "__main__":
    train_ml_model()