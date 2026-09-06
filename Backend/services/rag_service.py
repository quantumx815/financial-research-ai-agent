import os
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from services.embedding_service import generate_embedding


_CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data",
    "chroma",
)

_COLLECTION_NAME = "financial_documents"


@dataclass
class DocumentMetadata:
    company_symbol: str
    company_name: str
    document_type: str
    document_year: Optional[int]
    source: str
    source_url: Optional[str]
    document_id: str
    chunk_index: Optional[int] = None
    section: Optional[str] = None


@dataclass
class RetrievalResult:
    chunk_id: str
    chunk_text: str
    company_symbol: str
    document_type: str
    document_year: Optional[int]
    source: str
    source_url: Optional[str]
    document_id: str
    chunk_index: Optional[int]
    distance: Optional[float]


@dataclass
class RetrievalResponse:
    query: str
    company_symbol: str
    top_k: int
    results: List[RetrievalResult] = field(default_factory=list)
    total_found: int = 0
    error: Optional[str] = None


class RAGService:
    def __init__(self):
        os.makedirs(_CHROMA_DIR, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=_CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def get_collection(self):
        return self._collection

    def add_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> None:
        embedding = generate_embedding(text)

        self._collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "company_symbol": metadata.company_symbol.upper(),
                "company_name": metadata.company_name,
                "document_type": metadata.document_type,
                "document_year": metadata.document_year or 0,
                "source": metadata.source,
                "source_url": metadata.source_url or "",
                "document_id": metadata.document_id,
                "chunk_index": metadata.chunk_index or 0,
                "section": metadata.section or "",
            }],
            ids=[chunk_id],
        )

    def upsert_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> None:
        embedding = generate_embedding(text)

        self._collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "company_symbol": metadata.company_symbol.upper(),
                "company_name": metadata.company_name,
                "document_type": metadata.document_type,
                "document_year": metadata.document_year or 0,
                "source": metadata.source,
                "source_url": metadata.source_url or "",
                "document_id": metadata.document_id,
                "chunk_index": metadata.chunk_index or 0,
                "section": metadata.section or "",
            }],
            ids=[chunk_id],
        )

    def query(
        self,
        query_text: str,
        company_symbol: str,
        n_results: int = 5,
    ) -> RetrievalResponse:
        if not query_text or not query_text.strip():
            return RetrievalResponse(
                query=query_text or "",
                company_symbol=company_symbol.upper() if company_symbol else "",
                top_k=n_results,
                error="Query text is empty.",
            )

        if not company_symbol or not company_symbol.strip():
            return RetrievalResponse(
                query=query_text,
                company_symbol=company_symbol or "",
                top_k=n_results,
                error="Company symbol is empty.",
            )

        try:
            query_embedding = generate_embedding(query_text)
        except Exception as exc:
            return RetrievalResponse(
                query=query_text,
                company_symbol=company_symbol.upper(),
                top_k=n_results,
                error=f"Embedding generation failed: {exc}",
            )

        try:
            raw_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"company_symbol": company_symbol.upper()},
            )
        except Exception as exc:
            return RetrievalResponse(
                query=query_text,
                company_symbol=company_symbol.upper(),
                top_k=n_results,
                error=f"ChromaDB query failed: {exc}",
            )

        results: List[RetrievalResult] = []

        ids = raw_results.get("ids", [[]])[0] if raw_results.get("ids") else []
        documents = raw_results.get("documents", [[]])[0] if raw_results.get("documents") else []
        metadatas = raw_results.get("metadatas", [[]])[0] if raw_results.get("metadatas") else []
        distances = raw_results.get("distances", [[]])[0] if raw_results.get("distances") else []

        for idx in range(len(ids)):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            results.append(
                RetrievalResult(
                    chunk_id=ids[idx] if idx < len(ids) else "",
                    chunk_text=documents[idx] if idx < len(documents) else "",
                    company_symbol=metadata.get("company_symbol", company_symbol.upper()),
                    document_type=metadata.get("document_type", ""),
                    document_year=metadata.get("document_year") or None,
                    source=metadata.get("source", ""),
                    source_url=metadata.get("source_url") or None,
                    document_id=metadata.get("document_id", ""),
                    chunk_index=metadata.get("chunk_index") if metadata.get("chunk_index") is not None else None,
                    distance=distances[idx] if idx < len(distances) else None,
                )
            )

        return RetrievalResponse(
            query=query_text,
            company_symbol=company_symbol.upper(),
            top_k=n_results,
            results=results,
            total_found=len(results),
        )

    def list_companies(self) -> list[str]:
        results = self._collection.get()
        symbols = set()
        for meta in results.get("metadatas", []):
            if meta and "company_symbol" in meta:
                symbols.add(meta["company_symbol"])
        return sorted(symbols)

    def delete_company_documents(self, company_symbol: str) -> None:
        results = self._collection.get(
            where={"company_symbol": company_symbol.upper()}
        )
        if results and results.get("ids"):
            self._collection.delete(ids=results["ids"])


rag_service = RAGService()
