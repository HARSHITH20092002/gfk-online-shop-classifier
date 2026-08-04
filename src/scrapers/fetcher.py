import cloudscraper
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_website_content(url):
    """Fetches web content using full browser header emulation to bypass WAF blocks."""
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
        
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        }
    )
    
    # Advanced browser headers to pass modern anti-bot checks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }

    response = None
    try:
        response = scraper.get(url, headers=headers, timeout=10)
    except Exception:
        # Retry with HTTP if HTTPS fails
        if url.startswith("https://"):
            try:
                response = scraper.get(url.replace("https://", "http://"), headers=headers, timeout=10)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                return None

    if response is None or response.status_code != 200:
        status = response.status_code if response else "No Response"
        print(f"Notice: HTTP Status {status} for {url}")
        return None

    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract meta tags (critical for JS-heavy sites like Temu)
        meta_content = []
        for tag in soup.find_all('meta'):
            content = tag.get('content', '')
            name = tag.get('name', '').lower()
            prop = tag.get('property', '').lower()
            if content and (name in ['description', 'keywords'] or 'og:' in prop):
                meta_content.append(content)

        # Remove scripts and styles
        for element in soup(["script", "style", "noscript"]):
            element.extract()
            
        page_text = soup.get_text(separator=' ').lower()
        links = [a.get('href', '').lower() for a in soup.find_all('a') if a.get('href')]
        
        full_text = page_text + " " + " ".join(meta_content).lower()
        
        return {
            "raw_text": full_text,
            "links": links,
            "html_raw": str(soup).lower()
        }
    except Exception as parse_err:
        print(f"Parsing error for {url}: {parse_err}")
        return None