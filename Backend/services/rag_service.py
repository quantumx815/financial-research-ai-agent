import os
from dataclasses import dataclass
from typing import Optional

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
    ) -> dict:
        query_embedding = generate_embedding(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"company_symbol": company_symbol.upper()},
        )

        return results

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
