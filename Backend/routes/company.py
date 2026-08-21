from fastapi import APIRouter, HTTPException

from services.financial_data import get_company_data
from services.news_service import get_company_news
from services.gemini_service import generate_financial_analysis

router = APIRouter()


@router.get("/company/{symbol}")
def get_company(symbol: str):
    try:
        return get_company_data(symbol.upper())
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Company not found"
        )


@router.get("/company/{symbol}/news")
def get_company_news_route(symbol: str):
    try:
        news = get_company_news(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "news": news
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve company news"
        )


@router.get("/company/{symbol}/analysis")
def get_company_analysis(symbol: str):
    try:
        symbol = symbol.upper()

        company_data = get_company_data(symbol)
        news = get_company_news(symbol)

        analysis = generate_financial_analysis(
            company_data,
            news
        )

        return {
            "symbol": symbol,
            "analysis": analysis
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate financial analysis"
        )