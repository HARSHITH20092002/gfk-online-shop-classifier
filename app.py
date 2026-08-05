import streamlit as st
import pandas as pd
import time
from urllib.parse import urlparse
from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import DualShopClassifier

# Page Configuration
st.set_page_config(
    page_title="GfK Online Shop Classifier",
    page_icon="🛍️",
    layout="wide"
)

@st.cache_resource
def load_classifier():
    return DualShopClassifier()

classifier = load_classifier()

def extract_domain(url):
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme if parsed.scheme else "https"
        netloc = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        return f"{scheme}://{netloc}"
    except Exception:
        return url

st.title("🛍️ GfK Online Shop Classifier")
st.caption("Hybrid E-Commerce Detection Engine: Rule-Based Heuristics + Machine Learning")

# Navigation Sidebar
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Select Mode:", ["Single Domain Analysis", "Batch Dataset Processing"])

if mode == "Single Domain Analysis":
    st.subheader("Single Domain Inspection")
    url_input = st.text_input("Enter Web Domain / URL:", placeholder="https://www.example.com")

    if st.button("Classify Domain", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
        else:
            domain = extract_domain(url_input)
            with st.spinner(f"Fetching and analyzing content from {domain}..."):
                start_time = time.time()
                page_data = fetch_website_content(domain)
                result = classifier.predict(page_data)
                elapsed = round(time.time() - start_time, 2)

            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result["is_shop"]:
                    st.success("Result: **SHOP**")
                else:
                    st.error("Result: **NOT A SHOP**")

            with col2:
                st.metric("Confidence Score", f"{int(result['confidence'] * 100)}%")

            with col3:
                st.metric("Decision Engine", result["method"], delta=f"{elapsed}s")

            st.subheader("Raw Extraction Insights")
            if page_data:
                st.json({
                    "domain": domain,
                    "content_length": len(page_data.get("raw_text", "")),
                    "discovered_links": len(page_data.get("links", [])),
                    "snippet": page_data.get("raw_text", "")[:300] + "..."
                })
            else:
                st.warning("Could not extract live HTML content from this URL.")

elif mode == "Batch Dataset Processing":
    st.subheader("Batch File Classification")
    st.write("Upload a `.csv` or `.parquet` file containing a column named `url` or `domain`.")

    uploaded_file = st.file_uploader("Upload Dataset File", type=["csv", "parquet"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_parquet(uploaded_file)

            target_col = "url" if "url" in df.columns else ("domain" if "domain" in df.columns else None)

            if not target_col:
                st.error("The uploaded file must contain either a 'url' or 'domain' column.")
            else:
                st.success(f"Dataset loaded successfully with {len(df)} total rows.")
                sample_size = st.slider("Select maximum domains to analyze:", 5, min(100, len(df)), 10)

                if st.button("Run Batch Classification", type="primary"):
                    urls_to_process = df[target_col].dropna().unique()[:sample_size]
                    
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    start_time = time.time()

                    for idx, raw_url in enumerate(urls_to_process):
                        domain = extract_domain(raw_url)
                        status_text.text(f"[{idx+1}/{len(urls_to_process)}] Processing: {domain}")
                        
                        item_start = time.time()
                        page_data = fetch_website_content(domain)
                        pred = classifier.predict(page_data)
                        elapsed = round(time.time() - item_start, 2)

                        results.append({
                            "domain": domain,
                            "is_shop": pred["is_shop"],
                            "status": "SHOP" if pred["is_shop"] else "NOT A SHOP",
                            "confidence": pred["confidence"],
                            "method": pred["method"],
                            "latency_sec": elapsed
                        })

                        progress_bar.progress((idx + 1) / len(urls_to_process))

                    status_text.text("Batch processing complete!")
                    total_time = round(time.time() - start_time, 2)

                    res_df = pd.DataFrame(results)

                    st.divider()
                    st.subheader("Batch Summary Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    
                    shops_count = (res_df["is_shop"] == True).sum()
                    m1.metric("Total Domains", len(res_df))
                    m2.metric("Shops Identified", f"{shops_count} ({round(shops_count/len(res_df)*100, 1)}%)")
                    m3.metric("Avg Latency", f"{round(res_df['latency_sec'].mean(), 2)}s")
                    m4.metric("Total Processing Time", f"{total_time}s")

                    st.dataframe(res_df, use_container_width=True)

                    # CSV Export
                    csv_data = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Batch Results CSV",
                        data=csv_data,
                        file_name="batch_classification_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")