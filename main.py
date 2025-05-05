from fastapi import FastAPI
from pydantic import BaseModel
from newspaper import Article
from datetime import datetime
import os
import uvicorn

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM

# FastAPI instance
app = FastAPI()

# DeepSeek configuration
os.environ["OPENAI_API_KEY"] = "sk-xxx"  # Replace at deploy, NOT in code directly
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-reasoner"

# Request schema
class ArticleRequest(BaseModel):
    url: str
    country: str
    category: str

# Custom OpenAI wrapper for DeepSeek
llm = OpenAI(
    temperature=0,
    openai_api_base=DEEPSEEK_BASE_URL,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name=DEEPSEEK_MODEL
)

prompt = PromptTemplate(
    input_variables=["article"],
    template="""
You are a world-class news summarizer. Summarize the article below in under 200 words.

ARTICLE:
{article}

SUMMARY:
"""
)

chain = LLMChain(llm=llm, prompt=prompt)

@app.post("/process_url")
async def process_url(data: ArticleRequest):
    try:
        article = Article(data.url)
        article.download()
        article.parse()

        if not article.text.strip():
            return {"error": "No content extracted from article."}

        summary = chain.run(article=article.text)

        pub_date = article.publish_date
        if isinstance(pub_date, datetime):
            pub_date = pub_date.strftime('%Y-%m-%d')
        else:
            pub_date = str(datetime.utcnow().date())

        return {
            "url": data.url,
            "country": data.country,
            "category": data.category,
            "Headline": article.title,
            "Date": pub_date,
            "Summary": summary.strip()
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
