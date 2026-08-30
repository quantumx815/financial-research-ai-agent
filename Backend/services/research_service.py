from services.news_service import get_company_news


STRONG_FINANCIAL_KEYWORDS = {
    "earnings",
    "revenue",
    "profit",
    "loss",
    "guidance",
    "forecast",
    "outlook",
    "analyst",
    "rating",
    "upgrade",
    "downgrade",
    "investment",
    "investor",
    "acquisition",
    "merger",
    "deal",
    "regulation",
    "regulatory",
    "lawsuit",
    "legal",
    "fine",
    "antitrust",
    "dividend",
    "cash flow",
    "margin",
    "financial",
    "valuation",
    "ipo",
    "buyback",
    "insider",
    "sec",
    "earnings",
}


GENERAL_BUSINESS_KEYWORDS = {
    "product",
    "launch",
    "demand",
    "supply",
    "manufacturing",
    "quarter",
    "sales",
    "growth",
    "market",
    "stock",
    "shares",
    "partnership",
    "strategy",
    "expansion",
    "cost",
    "pricing",
    "revenue",
    "profit",
    "loss",
    "customer",
    "retail",
    "consumer",
    "technology",
    "ai",
    "artificial intelligence",
    "chip",
    "semiconductor",
    "energy",
    "oil",
    "gas",
    "renewable",
    "electric vehicle",
    "ev",
    "mobility",
    "software",
    "cloud",
    "data",
    "platform",
    "subscription",
    "advertising",
}


def _score_text(text: str) -> int:
    lowered = text.lower()
    score = 0

    for keyword in STRONG_FINANCIAL_KEYWORDS:
        if keyword in lowered:
            score += 2

    for keyword in GENERAL_BUSINESS_KEYWORDS:
        if keyword in lowered:
            score += 1

    return score


def _categorize_relevance(score: int) -> str:
    if score >= 6:
        return "high"

    if score >= 3:
        return "medium"

    return "low"


def build_research_package(company_data, news):
    company_section = {
        "symbol": company_data.get("symbol"),
        "company_name": company_data.get("company_name"),
        "sector": company_data.get("sector"),
        "industry": company_data.get("industry"),
    }

    financial_metrics_section = {
        "current_price": company_data.get("current_price"),
        "market_cap": company_data.get("market_cap"),
        "currency": company_data.get("currency"),
        "revenue": company_data.get("revenue"),
        "net_income": company_data.get("net_income"),
        "eps": company_data.get("eps"),
        "pe_ratio": company_data.get("pe_ratio"),
        "profit_margin": company_data.get("profit_margin"),
        "revenue_growth": company_data.get("revenue_growth"),
        "return_on_equity": company_data.get("return_on_equity"),
        "debt_to_equity": company_data.get("debt_to_equity"),
        "fifty_two_week_high": company_data.get("fifty_two_week_high"),
        "fifty_two_week_low": company_data.get("fifty_two_week_low"),
    }

    seen_titles = set()
    scored_news = []

    for item in news:
        title = (item.get("title") or "").strip()

        if not title:
            continue

        if title in seen_titles:
            continue

        seen_titles.add(title)

        summary = (item.get("summary") or "").strip()
        combined_text = f"{title} {summary}"
        relevance_score = _score_text(combined_text)
        relevance = _categorize_relevance(relevance_score)

        scored_news.append({
            "title": title,
            "summary": item.get("summary"),
            "publisher": item.get("publisher"),
            "published_at": item.get("published_at"),
            "url": item.get("url"),
            "relevance_score": relevance_score,
            "relevance": relevance,
        })

    scored_news.sort(key=lambda article: article["relevance_score"], reverse=True)

    return {
        "company": company_section,
        "financial_metrics": financial_metrics_section,
        "recent_news": scored_news,
    }
