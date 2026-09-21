import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from services.bge_embedding_service import generate_bge_embedding, get_bge_model_info, get_bge_dimension


_BGE_CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data",
    "chroma_bge",
)

_BGE_COLLECTION_NAME = "financial_documents_bge"


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
    embedding_model: Optional[str] = None
    section: Optional[str] = None


@dataclass
class RetrievalResponse:
    query: str
    company_symbol: str
    top_k: int
    results: List[RetrievalResult] = field(default_factory=list)
    total_found: int = 0
    error: Optional[str] = None


class BGERAGService:
    def __init__(self):
        os.makedirs(_BGE_CHROMA_DIR, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=_BGE_CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_BGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._model_info = get_bge_model_info()

    def get_collection(self):
        return self._collection

    def get_model_info(self) -> dict:
        return self._model_info.copy()

    def _build_metadata_dict(self, metadata: DocumentMetadata) -> dict:
        meta = {
            "company_symbol": metadata.company_symbol.upper(),
            "company_name": metadata.company_name,
            "document_type": metadata.document_type,
            "document_year": metadata.document_year or 0,
            "source": metadata.source,
            "source_url": metadata.source_url or "",
            "document_id": metadata.document_id,
            "chunk_index": metadata.chunk_index or 0,
            "section": metadata.section or "",
            "embedding_provider": self._model_info["embedding_provider"],
            "embedding_model": self._model_info["embedding_model"],
            "embedding_dimension": self._model_info["embedding_dimension"],
            "embedding_version": self._model_info["embedding_version"],
        }
        return meta

    def add_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> None:
        embedding = generate_bge_embedding(text)

        self._collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[self._build_metadata_dict(metadata)],
            ids=[chunk_id],
        )

    def upsert_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
        embedding: Optional[List[float]] = None,
    ) -> None:
        if embedding is None:
            embedding = generate_bge_embedding(text)

        self._collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[self._build_metadata_dict(metadata)],
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
            query_embedding = generate_bge_embedding(query_text)
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
            document_id = metadata.get("document_id", "")
            embedding_model = metadata.get("embedding_model", self._model_info["embedding_model"])

            results.append(
                RetrievalResult(
                    chunk_id=ids[idx] if idx < len(ids) else "",
                    chunk_text=documents[idx] if idx < len(documents) else "",
                    company_symbol=metadata.get("company_symbol", company_symbol.upper()),
                    document_type=metadata.get("document_type", ""),
                    document_year=metadata.get("document_year") or None,
                    source=metadata.get("source", ""),
                    source_url=metadata.get("source_url") or None,
                    document_id=document_id,
                    chunk_index=metadata.get("chunk_index") if metadata.get("chunk_index") is not None else None,
                    distance=distances[idx] if idx < len(distances) else None,
                    embedding_model=embedding_model,
                    section=metadata.get("section") or None,
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

    def count(self) -> int:
        return self._collection.count()

    def get_all_chunks(self) -> List[dict]:
        results = self._collection.get(include=["metadatas", "documents"])
        chunks = []
        for i in range(len(results.get("ids", []))):
            chunks.append({
                "chunk_id": results["ids"][i],
                "text": results["documents"][i] if results.get("documents") else "",
                "metadata": results["metadatas"][i] if results.get("metadatas") else {},
            })
        return chunks


bge_rag_service = BGERAGService()