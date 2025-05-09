import os
import requests
import asyncio
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from newspaper import Article
from transformers import pipeline
from langchain_core.documents import Document

app = FastAPI()

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "News extractor"

DEEPSEEK_URL = "https://api.deepseek.com/v1/"
DEEPSEEK_MODEL = "deepseek-chat"

# Fallback summarizer (load once)
fallback_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ArticleInput(BaseModel):
    url: HttpUrl

# Root route
@app.get("/")
async def root():
    return {"message": "News extractor is live"}

# Country/category inference
def infer_country_category(text: str):
    text_lower = text.lower()
    country = "Global"
    category = "General"
    if "india" in text_lower:
        country = "India"
    elif "us" in text_lower or "america" in text_lower:
        country = "USA"
    elif "china" in text_lower:
        country = "China"

    if "finance" in text_lower or "stock" in text_lower:
        category = "Finance"
    elif "tech" in text_lower or "software" in text_lower:
        category = "Technology"
    elif "sports" in text_lower:
        category = "Sports"

    return country, category

# DeepSeek summarizer
async def summarize_with_deepseek(text: str) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={"model": DEEPSEEK_MODEL, "messages": [
                    {"role": "user", "content": f"Summarize this news article:\n\n{text}"}
                ]}
            )
            result = response.json()
            if response.status_code == 200 and "choices" in result:
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"DeepSeek error: {str(e)}")
    return None  # fail gracefully

# BART fallback
def summarize_with_bart(text: str) -> str:
    try:
        chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summaries = [fallback_summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                     for chunk in chunks[:3]]
        return " ".join(summaries)
    except Exception as e:
        print(f"BART error: {str(e)}")
        return "Summary unavailable."

# Airtable saver
async def save_to_airtable(record: dict):
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
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.post(url, headers=headers, json=payload)
    except Exception as e:
        print(f"Airtable error: {str(e)}")
        return None

# Article processor
async def process_article(url: str):
    article = Article(url)
    article.download()
    article.parse()

    if not article.text.strip():
        return {"error": "No article text found."}

    text = article.text
    title = article.title.strip()
    date = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else datetime.utcnow().strftime("%Y-%m-%d")
    country, category = infer_country_category(text)

    summary = await summarize_with_deepseek(text)
    if not summary:
        summary = summarize_with_bart(text)

    result = {
        "url": url,
        "title": title,
        "date": date,
        "country": country,
        "category": category,
        "summary": summary
    }

    airtable_response = await save_to_airtable(result)
    status = airtable_response.status_code if airtable_response else "Airtable failed"

    return {
        "status": "success",
        "data": result,
        "airtable_status": status
    }

# Main route
@app.post("/process_url")
async def handle_url(payload: ArticleInput):
    return await process_article(str(payload.url))

# Local dev run
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
