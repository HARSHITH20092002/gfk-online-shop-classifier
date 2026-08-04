import os
import pandas as pd
from urllib.parse import urlparse
from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import DualShopClassifier

def extract_domain(url):
    """Extracts root protocol and domain from deep URLs."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme if parsed.scheme else "https"
        netloc = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        return f"{scheme}://{netloc}"
    except Exception:
        return url

def process_parquet_dataset(file_path, limit=10, output_csv="data/processed/classification_results.csv"):
    print(f"\n--- Loading Dataset: {file_path} ---")
    df = pd.read_parquet(file_path)
    
    # Extract unique domain roots
    raw_urls = df['url'].dropna().unique()
    domains = list(dict.fromkeys([extract_domain(u) for u in raw_urls]))[:limit]
    
    classifier = DualShopClassifier()
    results = []

    print(f"\n--- Classifying Top {len(domains)} Root Domains ---\n")
    for domain in domains:
        print(f"Fetching & Classifying: {domain}")
        page_data = fetch_website_content(domain)
        prediction = classifier.predict(page_data)
        
        status = "SHOP" if prediction["is_shop"] else "NOT A SHOP"
        results.append({
            "domain": domain,
            "is_shop": prediction["is_shop"],
            "confidence": prediction["confidence"],
            "method": prediction["method"]
        })
        print(f"  -> Result: {status} | Confidence: {prediction['confidence']} | Method: {prediction['method']}\n")
    
    # Create the data/processed directory if it does not exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"--- Processing Complete! Results saved to {output_csv} ---")

if __name__ == "__main__":
    dataset_path = "data/FR_online_market_discovery_2024.parquet"
    process_parquet_dataset(dataset_path, limit=10)