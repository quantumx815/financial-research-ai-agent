import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv
from google import genai

from services.bge_query_service import bge_query_service, BGEQueryResponse
from services.bge_rag_service import RetrievalResult

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@dataclass
class QAResponse:
    """Structured response for grounded financial Q&A."""
    symbol: str
    query: str
    answer: str
    evidence: List[dict] = field(default_factory=list)
    evidence_count: int = 0
    available: bool = False
    error: Optional[str] = None


def _format_evidence_for_prompt(results: List[RetrievalResult]) -> str:
    """Format retrieved SEC evidence for Gemini prompt."""
    if not results:
        return ""
    
    evidence_text = "SEC FILING EVIDENCE:\n"
    evidence_text += (
        "The following excerpts were retrieved from official SEC filings "
        "using semantic search. Answer the user's question based ONLY on this evidence.\n"
        "If the evidence does not contain sufficient information to answer the question, "
        "explicitly state that the available evidence is insufficient.\n"
        "Do not use external knowledge or make claims not supported by the provided excerpts.\n\n"
    )
    
    for idx, result in enumerate(results, 1):
        evidence_text += (
            f"[Evidence {idx}]\n"
            f"Document Type: {result.document_type}\n"
            f"Filing Year: {result.document_year}\n"
            f"Document ID: {result.document_id}\n"
            f"Section: {result.section or 'Not specified'}\n"
            f"Chunk Index: {result.chunk_index}\n"
            f"Source: {result.source}\n"
            f"Source URL: {result.source_url}\n"
            f"Text: {result.chunk_text}\n\n"
        )
    
    return evidence_text


def _build_qa_prompt(query: str, evidence_text: str) -> str:
    """Build the prompt for grounded financial Q&A."""
    return f"""
You are a financial research assistant. Answer the user's question based ONLY on the SEC filing evidence provided below.

User Question: {query}

{evidence_text}

INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the SEC filing evidence provided above.
2. If the evidence is insufficient to answer the question, clearly state: "The available SEC filing evidence is insufficient to answer this question."
3. Do not use external knowledge, general financial knowledge, or make claims not directly supported by the provided excerpts.
4. Cite the evidence by referencing the evidence numbers (e.g., "According to Evidence 1...").
5. Use cautious language: may, could, suggests, indicates, appears.
6. Do not use unsupported certainty: definitely, guaranteed, will certainly.
7. Do not provide investment recommendations or price targets.
8. Keep the answer focused and concise.

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown, no extra text.

{{
  "answer": "",
  "sufficient_evidence": true/false
}}
"""


def _parse_qa_response(raw_text: str) -> dict:
    """Parse Gemini response to extract answer and sufficiency flag."""
    cleaned = raw_text.strip()
    
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    
    return json.loads(cleaned)


def _format_evidence_metadata(results: List[RetrievalResult]) -> List[dict]:
    """Format evidence metadata for response."""
    evidence = []
    for idx, result in enumerate(results, 1):
        evidence.append({
            "evidence_number": idx,
            "chunk_id": result.chunk_id,
            "document_type": result.document_type,
            "document_year": result.document_year,
            "document_id": result.document_id,
            "section": result.section,
            "chunk_index": result.chunk_index,
            "source": result.source,
            "source_url": result.source_url,
            "distance": result.distance,
            "embedding_model": result.embedding_model,
        })
    return evidence


def answer_financial_question(
    symbol: str,
    query: str,
    top_k: int = 5,
) -> QAResponse:
    """
    Answer a financial research question using SEC evidence from the BGE knowledge base.
    
    Flow:
    1. Query BGE for relevant SEC chunks
    2. If no evidence available, return clear response
    3. If evidence exists, send to Gemini with grounded prompt
    4. Return structured response with answer and evidence metadata
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "MSFT")
        query: Natural-language financial research question
        top_k: Maximum number of evidence chunks to retrieve (default: 5)
        
    Returns:
        QAResponse with answer, evidence, and metadata
    """
    # Step 1: Retrieve evidence from BGE
    bge_response = bge_query_service.query(
        symbol=symbol,
        query=query,
        top_k=top_k,
    )
    
    if not bge_response.available:
        return QAResponse(
            symbol=symbol.upper().strip() if symbol else "",
            query=query,
            answer="",
            evidence=[],
            evidence_count=0,
            available=False,
            error=bge_response.error or "Company not available in BGE knowledge base",
        )
    
    if bge_response.error:
        return QAResponse(
            symbol=bge_response.symbol,
            query=query,
            answer="",
            evidence=[],
            evidence_count=0,
            available=True,
            error=bge_response.error,
        )
    
    results = bge_response.results
    evidence_count = len(results)
    
    if evidence_count == 0:
        return QAResponse(
            symbol=bge_response.symbol,
            query=query,
            answer="No relevant SEC filing evidence found for this query.",
            evidence=[],
            evidence_count=0,
            available=True,
            error=None,
        )
    
    # Step 2: Format evidence for Gemini
    evidence_text = _format_evidence_for_prompt(results)
    
    # Step 3: Call Gemini for grounded answer
    if not _GEMINI_CLIENT:
        return QAResponse(
            symbol=bge_response.symbol,
            query=query,
            answer="",
            evidence=_format_evidence_metadata(results),
            evidence_count=evidence_count,
            available=True,
            error="Gemini API not configured (missing GEMINI_API_KEY)",
        )
    
    prompt = _build_qa_prompt(query, evidence_text)
    
    try:
        response = _GEMINI_CLIENT.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        parsed = _parse_qa_response(response.text)
        answer = parsed.get("answer", "")
        sufficient = parsed.get("sufficient_evidence", False)
        
        if not sufficient:
            answer = "The available SEC filing evidence is insufficient to answer this question."
        
        return QAResponse(
            symbol=bge_response.symbol,
            query=query,
            answer=answer,
            evidence=_format_evidence_metadata(results),
            evidence_count=evidence_count,
            available=True,
            error=None,
        )
        
    except Exception as exc:
        return QAResponse(
            symbol=bge_response.symbol,
            query=query,
            answer="",
            evidence=_format_evidence_metadata(results),
            evidence_count=evidence_count,
            available=True,
            error=f"Gemini generation failed: {exc}",
        )


# Global instance
financial_qa_service = None  # Not a class, just use the function directly