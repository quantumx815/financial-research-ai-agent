import yfinance as yf


def get_company_data(ticker_symbol: str):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    return {
        "symbol": ticker_symbol.upper(),
        "company_name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "currency": info.get("currency"),
        "revenue": info.get("totalRevenue"),
        "net_income": info.get("netIncomeToCommon"),
        "eps": info.get("trailingEps"),
        "pe_ratio": info.get("trailingPE"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "return_on_equity": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }
