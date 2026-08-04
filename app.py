import streamlit as st
from src.scrapers.fetcher import fetch_website_content
from src.models.classifier import DualShopClassifier

st.set_page_config(page_title="GFK Online Shop Classifier", page_icon="🛍️", layout="centered")

st.title("🛍️ E-Commerce Shop Classifier")
st.markdown("Dual-solution intelligence pipeline combining heuristic rules and machine learning.")

@st.cache_resource
def load_classifier():
    return DualShopClassifier()

classifier = load_classifier()

url_input = st.text_input("Enter Website URL:", "https://www.temu.com")

if st.button("Classify Domain", type="primary"):
    if url_input:
        with st.spinner(f"Analyzing {url_input}..."):
            page_data = fetch_website_content(url_input)
            result = classifier.predict(page_data)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "SHOP 🛒" if result["is_shop"] else "NOT A SHOP 📄"
            st.metric("Classification", status)
        with col2:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        with col3:
            st.metric("Engine Used", result["method"])
            
        if page_data:
            with st.expander("Show Scraped Text Snippet"):
                st.text(page_data["raw_text"][:1000])