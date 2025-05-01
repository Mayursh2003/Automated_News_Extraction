from fastapi import FastAPI
from transformers import pipeline
from bs4 import BeautifulSoup
from pyairtable import Table
from datetime import datetime
import torch, os, requests

app = FastAPI()

# Load model on GPU if available
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Airtable config (use Railway variables in production)
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID", "app4TiP7ZblxV2Wyh")
TABLE_NAME = os.getenv("TABLE_NAME", "News extractor")
table = Table(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)

def extract_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        print(f"❌ Failed to fetch URL: {url} — {e}")
        return ""

def detect_country(text):
    countries = ["India", "USA", "UK", "China", "Germany"]
    return [c for c in countries if c.lower() in text.lower()]

def detect_category(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["trade", "tariff", "import"]): return "Trade"
    if any(w in text_lower for w in ["economy", "gdp", "inflation"]): return "Economy"
    if any(w in text_lower for w in ["election", "government", "policy"]): return "Politics"
    if any(w in text_lower for w in ["ai", "tech", "startup"]): return "Technology"
    if any(w in text_lower for w in ["cricket", "football", "olympics"]): return "Sports"
    return "Other"

def process_articles():
    records = table.all()
    for record in records:
        fields = record.get("fields", {})
        url = fields.get("URL", "")
        if url and not fields.get("Headline"):
            print(f"🔍 Processing: {url}")
            full_text = extract_article_text(url)
            if not full_text.strip():
                continue
            summary = summarizer(full_text[:1024], max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
            headline = summary.split('.')[0]
            country = detect_country(full_text)
            category = detect_category(full_text)
            table.update(record["id"], {
                "Headline": headline,
                "Date": datetime.utcnow().strftime('%Y-%m-%d'),
                "Country": country,
                "Category": category
            })
            print("✅ Record updated.")

@app.get("/")
def health_check():
    return {"message": "📰 News Summarizer API is running"}

@app.get("/run")
def run_summary():
    process_articles()
    return {"status": "✅ All articles processed"}
