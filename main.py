import os
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from newspaper import Article
from transformers import pipeline
from langchain_core.documents import Document



app = FastAPI()

# Environment Variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "News extractor"

DEEPSEEK_URL = "https://api.deepseek.com/v1/"
DEEPSEEK_MODEL = "deepseek-chat"

# Fallback summarizer
fallback_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class ArticleInput(BaseModel):
    url: HttpUrl

# Country/Category inference
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

# Airtable push
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

# DeepSeek summarizer
def summarize_with_deepseek(text: str) -> str:
    try:
        response = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"model": DEEPSEEK_MODEL, "messages": [
                {"role": "user", "content": f"Summarize this news article:\n\n{text}"}
            ]}
        )
        result = response.json()
        if response.status_code == 200 and "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception(result.get("error", {}).get("message", "DeepSeek error"))
    except Exception as e:
        print(f"DeepSeek failed: {str(e)}")
        raise

# BART summarizer fallback
def summarize_with_bart(text: str) -> str:
    try:
        chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summaries = [fallback_summarizer(chunk, max_length=150, min_length=30,do_sample=False)[0]['summary_text']
                     for chunk in chunks[:]]  # limit to first 3 chunks
        return " ".join(summaries)
    except Exception as e:
        print(f"BART summarization failed: {str(e)}")
        return "Summary unavailable."

@app.post("/process_url")
async def process_url(payload: ArticleInput):
    try:
        url = str(payload.url)

        # Step 1: Extract Article
        article = Article(url)
        article.download()
        article.parse()

        if not article.text.strip():
            return {"error": "Failed to extract article text."}

        text = article.text
        title = article.title.strip()
        date = article.publish_date.strftime("%Y-%m-%d") if article.publish_date else datetime.utcnow().strftime("%Y-%m-%d")
        country, category = infer_country_category(text)

        # Step 2: Try DeepSeek
        try:
            summary = summarize_with_deepseek(text)
        except:
            summary = summarize_with_bart(text)

        # Step 3: Save and Return
        result = {
            "url": url,
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
