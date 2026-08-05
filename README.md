# GfK Online Shop Classifier

An intelligent dual-solution system designed to discover, fetch, and classify web domains into **Shop** vs. **Non-Shop** categories using a hybrid approach combining heuristic web scraping rules and machine learning (TF-IDF + Logistic Regression).

---

## Key Features

- **Dual-Engine Architecture:**
  - **Solution 1 (Rule-Based Heuristic Engine):** Scans scraped web content for e-commerce indicators (e.g., cart keywords, checkout paths, payment provider signatures, product price structures).
  - **Solution 2 (Machine Learning Engine):** Uses TF-IDF n-gram vectorization on scraped page metadata, body text, and structural elements with a trained Logistic Regression model.
- **Interactive Web Dashboard:** Built with Streamlit for single-domain testing and real-time confidence visualizer.
- **Batch Benchmarking Suite:** Automated script to evaluate model accuracy, latency, and decision-engine breakdown across 50+ domains from GfK market discovery Parquet datasets.

---

## Project Architecture & Directory Structure

```text
gfk-online-shop-classifier/
├── data/                         # Dataset directory (.gitignore tracked)
│   └── processed/                # Saved benchmark outputs and metrics
├── src/
│   ├── models/
│   │   ├── classifier.py         # Dual-solution prediction engine
│   │   ├── train.py              # ML model training script
│   │   ├── logistic_model.pkl    # Serialized ML classifier
│   │   └── tfidf_vectorizer.pkl  # Serialized vectorizer
│   └── scrapers/
│       └── fetcher.py            # Web scraper with timeout & user-agent handling
├── app.py                        # Streamlit interactive web dashboard
├── benchmark.py                  # Batch evaluation on 50 domains
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation