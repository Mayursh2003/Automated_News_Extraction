import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import torch
from datetime import datetime
from pyairtable import Table
import os
import time

# Load summarizer model on CUDA
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Airtable setup (set these via environment variables or hardcode for local testing)
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY") or "pat9v6A8Gtgt7YJBW.1db41060c62cc46e2fa15841a9a2038b7e309306ddc4c32af81440942b603950"
BASE_ID = "app4TiP7ZblxV2Wyh"
TABLE_NAME = "News extractor"

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
    if any(w in text_lower for w in ["trade", "tariff", "import"]):
        return "Trade"
    if any(w in text_lower for w in ["economy", "gdp", "inflation"]):
        return "Economy"
    if any(w in text_lower for w in ["election", "government", "policy"]):
        return "Politics"
    if any(w in text_lower for w in ["ai", "tech", "startup"]):
        return "Technology"
    if any(w in text_lower for w in ["cricket", "football", "olympics"]):
        return "Sports"
    return "Other"

def process_pending_articles():
    records = table.all()
    for record in records:
        fields = record.get("fields", {})
        url = fields.get("URL", "")
        if url and not fields.get("Headline"):  # Only process incomplete entries
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
            print("✅ Record updated.\n")

# Run once or periodically
if __name__ == "__main__":
    print("🚀 Running article processor...")
    process_pending_articles()
