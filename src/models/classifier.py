from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DualShopClassifier:
    def __init__(self):
        # Solution 1: Explicit transactional markers & retail indicators
        self.strong_keywords = {
            'add to cart', 'add to basket', 'checkout', 'shopping cart', 'view cart', 
            'buy now', 'my cart', 'panier', 'commander', 'ajouter au panier', 
            'achat en ligne', 'mon panier', 'free shipping', 'livraison gratuite'
        }
        self.secondary_keywords = {
            'cart', 'basket', 'price', 'shop', 'store', 'boutique', 'vente', 'achat',
            'promotions', 'soldes', 'catalog', 'catalogue', 'retrait magasin', '$', '£', '€'
        }
        
        # Solution 2: Machine Learning Model Setup
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.ml_model = LogisticRegression()
        self._train_dummy_ml_baseline()

    def _train_dummy_ml_baseline(self):
        training_corpus = [
            # E-Commerce Samples (1)
            "buy products add to cart free shipping secure checkout price product item order summary delivery store catalog boutique livraison commander",
            "shop online discount store items add to basket sale add to cart fast shipping categories catalog buy now cart achat en ligne promo boutique",
            "electronics store buy online add to cart view cart checkout currency deals discount items search product outillage brico matériel vente prix",
            "temu online shopping deals discounts fashion home electronics free shipping savings shop now",
            
            # Non-Shop Samples (0)
            "latest news articles blog post daily update editorial contact us privacy policy terms search main page read article actualités publication",
            "company profile global services team about us leadership career investors media release overview corporate details entreprise groupe",
            "wikipedia encyclopedia article references external links edit page free content history repository study main page community portal livre ebook viewer PDF document reader online reader publication view book translate google search tool dictionary job careers application hire employee",
            "academic research network publication paper citations author profile researchgate science journal"
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        X = self.vectorizer.fit_transform(training_corpus)
        self.ml_model.fit(X, labels)

    def solution_1_heuristics(self, data):
        text = data["raw_text"]
        links = " ".join(data["links"])
        full_content = f"{text} {links}"
        
        strong_matches = [kw for kw in self.strong_keywords if kw in full_content]
        secondary_matches = [kw for kw in self.secondary_keywords if kw in full_content]
        
        if len(strong_matches) >= 1 or len(secondary_matches) >= 2:
            return {"is_shop": True, "confidence": 0.95, "method": "Solution 1 (Heuristic)"}
            
        return None

    def solution_2_ml(self, text):
        """Machine Learning engine with strict 0.70 confidence threshold."""
        X = self.vectorizer.transform([text])
        shop_proba = float(self.ml_model.predict_proba(X)[0][1])  # Probability of class 1 (SHOP)
        
        # Enforce minimum 70% probability threshold for positive shop classification
        if shop_proba >= 0.70:
            return {
                "is_shop": True,
                "confidence": round(shop_proba, 2),
                "method": "Solution 2 (ML Engine)"
            }
        else:
            return {
                "is_shop": False,
                "confidence": round(1.0 - shop_proba, 2),
                "method": "Solution 2 (ML Engine)"
            }

    def predict(self, page_data):
        if not page_data:
            return {"is_shop": False, "confidence": 0.0, "method": "Failed Fetch"}
            
        h_result = self.solution_1_heuristics(page_data)
        if h_result:
            return h_result
            
        return self.solution_2_ml(page_data["raw_text"])