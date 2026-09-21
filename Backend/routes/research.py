from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.financial_qa_service import answer_financial_question

router = APIRouter()


class ResearchQueryRequest(BaseModel):
    symbol: str = Field(..., min_length=1, description="Stock symbol (e.g., AAPL, MSFT)")
    query: str = Field(..., min_length=1, description="Natural-language financial research question")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum number of evidence chunks to retrieve")


class EvidenceMetadata(BaseModel):
    evidence_number: int
    chunk_id: str
    document_type: str
    document_year: int
    document_id: str
    section: str | None = None
    chunk_index: int
    source: str
    source_url: str
    distance: float
    embedding_model: str


class ResearchQueryResponse(BaseModel):
    symbol: str
    query: str
    answer: str
    evidence: list[EvidenceMetadata] = []
    evidence_count: int
    available: bool
    error: str | None = None


@router.post("/research/query")
def query_financial_research(request: ResearchQueryRequest) -> ResearchQueryResponse:
    """
    Answer a financial research question using SEC filing evidence from the BGE knowledge base.
    
    This endpoint performs grounded Q&A:
    1. Retrieves relevant SEC filing chunks using local BGE embeddings
    2. Sends evidence + question to Gemini for grounded answer generation
    3. Returns structured response with answer and evidence metadata
    """
    try:
        qa_response = answer_financial_question(
            symbol=request.symbol.strip(),
            query=request.query.strip(),
            top_k=request.top_k,
        )
        
        # If company not available, return 404
        if not qa_response.available and qa_response.error:
            if "not indexed" in qa_response.error.lower() or "not available" in qa_response.error.lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Company '{request.symbol.upper()}' not available in research knowledge base"
                )
            raise HTTPException(
                status_code=400,
                detail=qa_response.error
            )
        
        # If available but has error AND no evidence, it's a service error - return 400
        if qa_response.available and qa_response.error and qa_response.evidence_count == 0:
            raise HTTPException(
                status_code=400,
                detail=qa_response.error
            )
        
        # Otherwise return 200 with response (evidence may be empty or populated, error may be present)
        evidence_models = [EvidenceMetadata(**ev) for ev in qa_response.evidence]
        
        return ResearchQueryResponse(
            symbol=qa_response.symbol,
            query=qa_response.query,
            answer=qa_response.answer,
            evidence=evidence_models,
            evidence_count=qa_response.evidence_count,
            available=qa_response.available,
            error=qa_response.error,
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal server error during financial research query"
        ) from exc