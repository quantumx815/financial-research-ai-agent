import os
from dataclasses import dataclass, field, asdict
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

    def _build_metadata_dict(self, metadata: DocumentMetadata, generation: int = 1, supersedes_chunk_id: Optional[str] = None) -> dict:
        """Build metadata dict for ChromaDB, including optional migration fields."""
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
        }
        if generation != 1:
            meta["generation"] = generation
        if supersedes_chunk_id:
            meta["supersedes_chunk_id"] = supersedes_chunk_id
        return meta

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
            metadatas=[self._build_metadata_dict(metadata)],
            ids=[chunk_id],
        )

    def upsert_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
        generation: int = 1,
        supersedes_chunk_id: Optional[str] = None,
    ) -> None:
        embedding = generate_embedding(text)

        self._collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[self._build_metadata_dict(metadata, generation, supersedes_chunk_id)],
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
                n_results=n_results * 2,  # Fetch more to allow generation filtering
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

        # Build list of candidate results with generation info
        candidates = []
        for idx in range(len(ids)):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            generation = metadata.get("generation", 1)
            document_id = metadata.get("document_id", "")
            candidates.append({
                "chunk_id": ids[idx] if idx < len(ids) else "",
                "chunk_text": documents[idx] if idx < len(documents) else "",
                "company_symbol": metadata.get("company_symbol", company_symbol.upper()),
                "document_type": metadata.get("document_type", ""),
                "document_year": metadata.get("document_year") or None,
                "source": metadata.get("source", ""),
                "source_url": metadata.get("source_url") or None,
                "document_id": document_id,
                "chunk_index": metadata.get("chunk_index") if metadata.get("chunk_index") is not None else None,
                "generation": generation,
                "distance": distances[idx] if idx < len(distances) else None,
            })

        # Group by document_id, keep only highest generation per document
        doc_to_max_gen = {}
        for c in candidates:
            doc_id = c["document_id"]
            gen = c["generation"]
            if doc_id not in doc_to_max_gen or gen > doc_to_max_gen[doc_id]:
                doc_to_max_gen[doc_id] = gen

        # Filter: only keep chunks at max generation for their document
        filtered = [c for c in candidates if c["generation"] == doc_to_max_gen.get(c["document_id"], 1)]

        # Sort by distance (ascending) and take top n_results
        filtered.sort(key=lambda x: x["distance"] if x["distance"] is not None else float("inf"))
        filtered = filtered[:n_results]

        for c in filtered:
            results.append(
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    chunk_text=c["chunk_text"],
                    company_symbol=c["company_symbol"],
                    document_type=c["document_type"],
                    document_year=c["document_year"],
                    source=c["source"],
                    source_url=c["source_url"],
                    document_id=c["document_id"],
                    chunk_index=c["chunk_index"],
                    distance=c["distance"],
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

    def get_vector_by_id(self, chunk_id: str) -> Optional[dict]:
        """
        Retrieve an existing vector and its metadata from ChromaDB by chunk_id.
        
        Returns:
            Dict with 'embedding', 'document', 'metadata' keys, or None if not found.
        """
        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"]
            )
            if not result.get("ids") or len(result["ids"]) == 0:
                return None
            
            embeddings = result.get("embeddings", [])
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])
            
            # Handle numpy arrays from ChromaDB - check if embeddings exist and have content
            if embeddings is None or len(embeddings) == 0:
                return None
            # embeddings[0] could be a numpy array; check it's not None and has elements
            first_emb = embeddings[0]
            if first_emb is None:
                return None
            # Convert to list if it's a numpy array
            try:
                if hasattr(first_emb, 'tolist'):
                    embedding_list = first_emb.tolist()
                else:
                    embedding_list = list(first_emb)
            except Exception:
                return None
            if not embedding_list:
                return None
            
            return {
                "embedding": embedding_list,
                "document": documents[0] if documents and len(documents) > 0 else "",
                "metadata": metadatas[0] if metadatas and len(metadatas) > 0 else {},
            }
        except Exception:
            return None

    def upsert_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: DocumentMetadata,
        generation: int = 1,
        supersedes_chunk_id: Optional[str] = None,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """
        Upsert a document chunk, optionally using a pre-computed embedding.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: Chunk text content
            metadata: Document metadata
            generation: Generation number (1 for original, 2 for migration)
            supersedes_chunk_id: Old chunk ID this chunk supersedes
            embedding: Pre-computed embedding vector. If None, generates via Gemini.
        """
        if embedding is None:
            embedding = generate_embedding(text)

        self._collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[self._build_metadata_dict(metadata, generation, supersedes_chunk_id)],
            ids=[chunk_id],
        )


rag_service = RAGService()
