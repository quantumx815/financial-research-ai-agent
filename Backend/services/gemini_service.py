import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_financial_analysis(company_data, news):
    company_summary = {
        "symbol": company_data.get("symbol"),
        "company_name": company_data.get("company_name"),
        "sector": company_data.get("sector"),
        "industry": company_data.get("industry"),
        "current_price": company_data.get("current_price"),
        "market_cap": company_data.get("market_cap"),
        "currency": company_data.get("currency"),
    }

    news_summary = []

    for item in news[:3]:
        news_summary.append({
            "title": item.get("title"),
            "summary": item.get("summary"),
            "publisher": item.get("publisher"),
            "published_at": item.get("published_at"),
        })

    prompt = f"""
You are a financial research assistant.

Analyze the following company information and recent news.

Company Information:
{company_summary}

Recent News:
{news_summary}

Provide a concise financial research summary covering:

1. Company overview
2. Current financial position
3. Important recent news
4. Potential positive factors
5. Potential risks
6. Overall research perspective

Keep the response concise and useful for a financial research dashboard.

Do not give direct buy or sell recommendations.
Do not provide investment advice.
Clearly state that this is informational research, not financial advice.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text