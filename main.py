import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from newspaper import Article
from langchain_community.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

# Environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = "News extractor"  # Change if needed

# DeepSeek API URL
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-reasoner"

# FastAPI request schema
@app.get("/")
def root():
    return {"message": "API is running!"}

class ArticleRequest(BaseModel):
    url: str
    country: str
    category: str

def save_to_airtable(data: dict):
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "fields": {
            "URL": data["url"],
            "Country": data["country"],
            "Category": data["category"],
            "Headline": data["Headline"],
            "Date": data["Date"],
            "Summary": data["Summary"]
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.status_code, response.text

@app.post("/process_url")
async def process_url(data: ArticleRequest):
    try:
        article = Article(data.url)
        article.download()
        article.parse()

        if not article.text.strip():
            return {"error": "No content extracted from article."}

        # Use DeepSeek to summarize
        deepseek_response = requests.post(
            f"{DEEPSEEK_BASE_URL}/summarize",
            json={"model": DEEPSEEK_MODEL, "text": article.text},
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        )
        deepseek_data = deepseek_response.json()

        # Extract summary from DeepSeek response
        summary = deepseek_data.get('summary', 'No summary available.')

        # Format publication date
        pub_date = article.publish_date
        if isinstance(pub_date, datetime):
            pub_date = pub_date.strftime('%Y-%m-%d')
        else:
            pub_date = str(datetime.utcnow().date())

        result = {
            "url": data.url,
            "country": data.country,
            "category": data.category,
            "Headline": article.title,
            "Date": pub_date,
            "Summary": summary.strip()
        }

        # Save data to Airtable
        save_to_airtable(result)

        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
