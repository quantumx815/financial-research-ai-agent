from fastapi import APIRouter, HTTPException

from services.financial_data import get_company_data
from services.news_service import get_company_news
from services.gemini_service import generate_financial_analysis
from services.rag_service import rag_service

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

        rag_context = None
        try:
            query_text = (
                f"What are {symbol}'s major financial risks, "
                f"business risks, opportunities, and financial performance?"
            )
            rag_response = rag_service.query(query_text, symbol, n_results=5)
            if rag_response.error is None and rag_response.results:
                rag_context = rag_response
        except Exception:
            rag_context = None

        analysis = generate_financial_analysis(
            company_data,
            news,
            rag_context=rag_context
        )

        response = {
            "symbol": symbol,
            "analysis": analysis
        }

        if rag_context and rag_context.error is None and rag_context.results:
            response["sec_evidence"] = [
                {
                    "document_type": result.document_type,
                    "document_year": result.document_year,
                    "document_id": result.document_id,
                    "chunk_index": result.chunk_index,
                    "source": result.source,
                    "source_url": result.source_url,
                    "chunk_text": result.chunk_text,
                }
                for result in rag_context.results
            ]

        return response

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate financial analysis"
        )