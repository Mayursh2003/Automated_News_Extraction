import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from newspaper import Article

app = FastAPI()

# Environment Variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "News extractor"

DEEPSEEK_URL = "https://api.deepseek.com/v1/summarize"
DEEPSEEK_MODEL = "deepseek-reasoner"

# Request schema
class ArticleInput(BaseModel):
    url: str

@app.get("/")
def root():
    return {"message": "FastAPI is live 🚀"}

# Helper to infer country and category
def infer_country_category(text: str):
    text_lower = text.lower()
    country = "Global"
    category = "General"

    if "india" in text_lower:
        country = "India"
    elif "us" in text_lower or "america" in text_lower:
        country = "USA"

    if "finance" in text_lower or "stock" in text_lower:
        category = "Finance"
    elif "tech" in text_lower or "software" in text_lower:
        category = "Technology"
    elif "sports" in text_lower:
        category = "Sports"

    return country, category

# Helper to save to Airtable
def save_to_airtable(record: dict):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "fields": {
            "URL": record["url"],
            "Headline": record["title"],
            "Date": record["date"],
            "Country": record["country"],
            "Category": record["category"],
            "Summary": record["summary"]
        }
    }
    return requests.post(url, headers=headers, json=payload)

@app.post("/process_url")
def process_url(payload: ArticleInput):
    try:
        article = Article(payload.url)
        article.download()
        article.parse()

        if not article.text.strip():
            return {"error": "Failed to extract article text."}

        # Auto-extract
        text = article.text
        title = article.title.strip()
        date = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else datetime.utcnow().strftime("%Y-%m-%d")
        country, category = infer_country_category(text)

        # Call DeepSeek to summarize
        deepseek_response = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"model": DEEPSEEK_MODEL, "text": text}
        )

        if deepseek_response.status_code != 200:
            return {"error": "DeepSeek summarization failed."}

        summary = deepseek_response.json().get("summary", "").strip()

        result = {
            "url": payload.url,
            "title": title,
            "date": date,
            "country": country,
            "category": category,
            "summary": summary
        }

        airtable_response = save_to_airtable(result)

        return {
            "status": "success",
            "data": result,
            "airtable_status": airtable_response.status_code
        }

    except Exception as e:
        return {"error": str(e)}
