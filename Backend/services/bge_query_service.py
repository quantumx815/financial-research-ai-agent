from dataclasses import dataclass, field
from typing import List, Optional

from services.bge_rag_service import bge_rag_service, RetrievalResult, RetrievalResponse
from services.company_availability_service import company_availability_service, CompanyAvailability


@dataclass
class BGEQueryResponse:
    """Structured response for BGE knowledge base queries."""
    symbol: str
    query: str
    top_k: int
    available: bool
    results: List[RetrievalResult] = field(default_factory=list)
    total_found: int = 0
    error: Optional[str] = None


class BGEQueryService:
    """
    Backend RAG query capability for the local BGE knowledge base.
    
    Provides a clean interface for retrieving SEC evidence from the
    local BGE Chroma collection with company filtering.
    """
    
    def __init__(self):
        self._bge_service = bge_rag_service
        self._avail_service = company_availability_service
    
    def query(
        self,
        symbol: str,
        query: str,
        top_k: int = 5,
    ) -> BGEQueryResponse:
        """
        Retrieve relevant SEC evidence from the local BGE knowledge base.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT")
            query: Natural-language financial research question
            top_k: Maximum number of results to return (default: 5)
            
        Returns:
            BGEQueryResponse with retrieval results and availability info
        """
        normalized_symbol = symbol.upper().strip() if symbol else ""
        
        if not normalized_symbol:
            return BGEQueryResponse(
                symbol=symbol or "",
                query=query,
                top_k=top_k,
                available=False,
                error="Invalid or empty symbol",
            )
        
        if not query or not query.strip():
            return BGEQueryResponse(
                symbol=normalized_symbol,
                query=query,
                top_k=top_k,
                available=False,
                error="Query text is empty.",
            )
        
        # Check company availability
        availability = self._avail_service.check_availability(normalized_symbol)
        
        if not availability.available:
            return BGEQueryResponse(
                symbol=normalized_symbol,
                query=query,
                top_k=top_k,
                available=False,
                error=f"Company {normalized_symbol} not indexed in BGE knowledge base. Fresh research required.",
            )
        
        # Query BGE collection
        retrieval_response = self._bge_service.query(
            query_text=query,
            company_symbol=normalized_symbol,
            n_results=top_k,
        )
        
        if retrieval_response.error:
            return BGEQueryResponse(
                symbol=normalized_symbol,
                query=query,
                top_k=top_k,
                available=True,
                error=retrieval_response.error,
            )
        
        return BGEQueryResponse(
            symbol=normalized_symbol,
            query=query,
            top_k=top_k,
            available=True,
            results=retrieval_response.results,
            total_found=retrieval_response.total_found,
        )


bge_query_service = BGEQueryService()