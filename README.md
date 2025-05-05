# 📰 News Extractor MVP

This project is a Minimum Viable Product (MVP) that extracts and summarizes news articles using Python, stores structured data in Airtable, and integrates with a frontend interface like Softr or Glide. Built for automated news analysis using AI.

---

## 🚀 Features

- Extracts article text from a given URL
- Summarizes content using a transformer model (`facebook/bart-large-cnn`)
- Detects:
  - Headline
  - Country (from a preset list)
  - Category (e.g., Trade, Economy, Politics, Technology, Sports)
- Auto-fills fields in Airtable:
  - `URL`, `Headline`, `Date`, `Country`, `Category`
- Fully supports CUDA (GPU acceleration)
- Works with Softr or Glide as a user-facing interface

---

## 🛠 Requirements

- Python 3.8+
- HuggingFace Transformers
- BeautifulSoup
- Requests
- PyAirtable
- Torch

Install dependencies with:

```bash
pip install -r requirements.txt

Install the new package:

pip install -U langchain-community
