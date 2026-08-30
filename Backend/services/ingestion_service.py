import json
import os
import time
from typing import List, Optional

from services.rag_service import RAGService, DocumentMetadata
from services.embedding_service import generate_embedding


_rag_service = RAGService()

_DEFAULT_BATCH_SIZE = 10
_DEFAULT_BATCH_DELAY_SECONDS = 1.0


class IngestionStats:
    def __init__(self):
        self.total_chunks = 0
        self.embedded_chunks = 0
        self.upserted_chunks = 0
        self.failed_chunks = 0
        self.skipped_existing = 0
        self.errors: List[str] = []


def _load_processed_document(processed_path: str) -> Optional[dict]:
    if not os.path.exists(processed_path):
        return None

    try:
        with open(processed_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _chunk_exists(chunk_id: str) -> bool:
    try:
        result = _rag_service.get_collection().get(ids=[chunk_id])
        return bool(result.get("ids") and len(result["ids"]) > 0)
    except Exception:
        return False


def ingest_processed_document(
    processed_path: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_delay: float = _DEFAULT_BATCH_DELAY_SECONDS,
) -> IngestionStats:
    stats = IngestionStats()
    data = _load_processed_document(processed_path)

    if not data:
        stats.errors.append(f"Failed to load processed document: {processed_path}")
        return stats

    chunks = data.get("chunks", [])
    stats.total_chunks = len(chunks)

    if not chunks:
        return stats

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text", "")
        company_symbol = chunk.get("company_symbol", "")
        company_name = chunk.get("company_name", "")
        document_type = chunk.get("document_type", "")
        document_year = chunk.get("document_year")
        source = chunk.get("source", "")
        source_url = chunk.get("source_url", "")
        document_id = chunk.get("document_id", "")
        chunk_index = chunk.get("chunk_index", idx)
        section = chunk.get("section")

        if not chunk_id or not text:
            stats.failed_chunks += 1
            stats.errors.append(f"Missing chunk_id or text at index {idx} in {processed_path}")
            continue

        if _chunk_exists(chunk_id):
            stats.skipped_existing += 1
            continue

        try:
            embedding = generate_embedding(text)
            stats.embedded_chunks += 1
        except Exception as exc:
            stats.failed_chunks += 1
            stats.errors.append(f"Embedding failed for chunk {chunk_id}: {exc}")
            continue

        try:
            metadata = DocumentMetadata(
                company_symbol=company_symbol,
                company_name=company_name,
                document_type=document_type,
                document_year=document_year,
                source=source,
                source_url=source_url,
                document_id=document_id,
                chunk_index=chunk_index,
                section=section,
            )

            _rag_service.upsert_document_chunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
            )
            stats.upserted_chunks += 1
        except Exception as exc:
            stats.failed_chunks += 1
            stats.errors.append(f"Upsert failed for chunk {chunk_id}: {exc}")

        if (idx + 1) % batch_size == 0 and idx + 1 < len(chunks):
            time.sleep(batch_delay)

    return stats


def ingest_all_processed_documents(
    processed_dir: Optional[str] = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_delay: float = _DEFAULT_BATCH_DELAY_SECONDS,
) -> IngestionStats:
    if processed_dir is None:
        processed_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Data",
            "financial_documents",
            "processed",
        )

    aggregate = IngestionStats()

    for root, _, files in os.walk(processed_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue

            processed_path = os.path.join(root, filename)
            stats = ingest_processed_document(
                processed_path=processed_path,
                batch_size=batch_size,
                batch_delay=batch_delay,
            )

            aggregate.total_chunks += stats.total_chunks
            aggregate.embedded_chunks += stats.embedded_chunks
            aggregate.upserted_chunks += stats.upserted_chunks
            aggregate.failed_chunks += stats.failed_chunks
            aggregate.skipped_existing += stats.skipped_existing
            aggregate.errors.extend(stats.errors)

    return aggregate
