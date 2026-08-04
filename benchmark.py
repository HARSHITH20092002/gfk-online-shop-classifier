import os
import time
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

def run_benchmark(dataset_path, limit=50, output_csv="data/processed/benchmark_results.csv"):
    start_time = time.time()
    print("\n==================================================")
    print(f"   STARTING BENCHMARK RUN ({limit} DOMAINS)")
    print(f"   Dataset: {dataset_path}")
    print("==================================================\n")

    # Load Parquet dataset
    df = pd.read_parquet(dataset_path)
    raw_urls = df['url'].dropna().unique()
    domains = list(dict.fromkeys([extract_domain(u) for u in raw_urls]))[:limit]

    classifier = DualShopClassifier()
    results = []

    print(f"Extracted {len(domains)} unique root domains for testing.\n")

    for idx, domain in enumerate(domains, 1):
        print(f"[{idx}/{len(domains)}] Fetching & Classifying: {domain}")
        item_start = time.time()
        
        page_data = fetch_website_content(domain)
        prediction = classifier.predict(page_data)
        elapsed = round(time.time() - item_start, 2)

        status = "SHOP" if prediction["is_shop"] else "NOT A SHOP"
        
        results.append({
            "domain": domain,
            "is_shop": prediction["is_shop"],
            "status": status,
            "confidence": prediction["confidence"],
            "method": prediction["method"],
            "response_time_sec": elapsed
        })
        
        print(f"   -> Result: {status} | Conf: {prediction['confidence']} | Method: {prediction['method']} | Time: {elapsed}s\n")

    results_df = pd.DataFrame(results)

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    # Summary Metrics Calculation
    total_time = round(time.time() - start_time, 2)
    total_domains = len(results_df)
    shop_count = int((results_df["is_shop"] == True).sum())
    non_shop_count = int((results_df["is_shop"] == False).sum())
    avg_conf = round(results_df["confidence"].mean(), 2)
    avg_time = round(results_df["response_time_sec"].mean(), 2)

    method_counts = results_df["method"].value_counts().to_dict()

    print("\n==================================================")
    print("          BENCHMARK SUMMARY METRICS")
    print("==================================================")
    print(f"Total Domains Evaluated : {total_domains}")
    print(f"Total Time Elapsed      : {total_time}s (Avg {avg_time}s / domain)")
    print(f"Shops Detected          : {shop_count} ({round(shop_count/total_domains*100, 1)}%)")
    print(f"Non-Shops / Failed      : {non_shop_count} ({round(non_shop_count/total_domains*100, 1)}%)")
    print(f"Average Confidence      : {avg_conf}")
    print("\nDecision Engine Breakdown:")
    for method, count in method_counts.items():
        percentage = round(count / total_domains * 100, 1)
        print(f"  - {method}: {count} ({percentage}%)")
    print("==================================================")
    print(f"Full output saved to: {output_csv}\n")

if __name__ == "__main__":
    dataset = "data/FR_online_market_discovery_2024.parquet"
    run_benchmark(dataset, limit=50)