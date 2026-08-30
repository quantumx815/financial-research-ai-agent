import yfinance as yf


def get_company_news(symbol: str, count: int = 10):
    ticker = yf.Ticker(symbol.upper())

    news = ticker.get_news(count=count)

    results = []

    for item in news:
        content = item.get("content", {})

        results.append({
            "title": content.get("title"),
            "summary": content.get("summary"),
            "publisher": content.get("provider", {}).get("displayName"),
            "published_at": content.get("pubDate"),
            "url": content.get("canonicalUrl", {}).get("url"),
        })

    return results
