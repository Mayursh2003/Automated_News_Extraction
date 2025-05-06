import os
import requests
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from newspaper import Article
from time import sleep

app = FastAPI()

# Environment Variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "News extractor"

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Constants
MAX_RETRIES = 2
MAX_CHARS_FOR_SUMMARY = 4000

# Request Schema
class ArticleInput(BaseModel):
    url: HttpUrl  # Ensures URL is valid

# Helper: Infer Country & Category
def infer_country_category(text: str):
    text_lower = text.lower()
    country = "Global"
    category = "General"

    if "india" in text_lower:
        country = "India"
    elif "us" in text_lower or "america" in text_lower:
        country = "USA"
    elif "uk" in text_lower or "britain" in text_lower:
        country = "UK"

    if "finance" in text_lower or "stock" in text_lower:
        category = "Finance"
    elif "tech" in text_lower or "ai" in text_lower or "software" in text_lower:
        category = "Technology"
    elif "sports" in text_lower:
        category = "Sports"
    elif "health" in text_lower or "medicine" in text_lower:
        category = "Health"

    return country, category

# Helper: Call DeepSeek API with retry logic
def get_summary_from_deepseek(text: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        "Summarize the following news article in 4-5 concise sentences:\n\n" + text
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    for attempt in range(MAX_RETRIES):
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
        if response.status_code == 200:
            try:
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass
        sleep(1)  # wait before retrying
    raise Exception("DeepSeek summarization failed after retries.")

# Helper: Push data to Airtable
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

# Main Route
@app.post("/process_url")
def process_url(payload: ArticleInput):
    try:
        # Validate and parse article
        article = Article(payload.url)
        article.download()
        article.parse()

        if not article.text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract article text.")

        # Extract info
        text = article.text[:MAX_CHARS_FOR_SUMMARY]
        title = article.title.strip()
        date = (
            article.publish_date.strftime("%Y-%m-%d")
            if article.publish_date
            else datetime.utcnow().strftime("%Y-%m-%d")
        )
        country, category = infer_country_category(text)

        # Summarize with DeepSeek
        summary = get_summary_from_deepseek(text)

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
