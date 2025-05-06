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

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Request schema
class ArticleInput(BaseModel):
    url: str

# Infer country & category
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

# Save to Airtable
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

        text = article.text.strip()[:4000]  # Limit to ~4000 chars
        title = article.title.strip()
        date = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else datetime.utcnow().strftime("%Y-%m-%d")
        country, category = infer_country_category(text)

        # DeepSeek summarization
        deepseek_response = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert news summarizer."},
                    {"role": "user", "content": f"Summarize the following article in 4-5 lines:\n\n{text}"}
                ]
            }
        )

        if deepseek_response.status_code != 200:
            return {"error": "DeepSeek summarization failed.", "details": deepseek_response.text}

        summary = deepseek_response.json()['choices'][0]['message']['content'].strip()

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
