import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Dict

from services.rag_service import RAGService, DocumentMetadata
from services.embedding_service import generate_embedding, QuotaExhaustedError


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
        self.quota_exhausted_chunks = 0
        self.other_failed_chunks = 0
        self.errors: List[str] = []
        self.quota_exhausted_errors: List[str] = []
        self.other_errors: List[str] = []

    def add_error(self, message: str, is_quota_exhausted: bool = False):
        self.errors.append(message)
        if is_quota_exhausted:
            self.quota_exhausted_errors.append(message)
            self.quota_exhausted_chunks += 1
        else:
            self.other_errors.append(message)
            self.other_failed_chunks += 1
        self.failed_chunks += 1


class MigrationStats:
    """Statistics for migration operations."""
    def __init__(self):
        self.total_chunks = 0
        self.migrated_chunks = 0  # Successfully migrated to new generation
        self.skipped_unchanged = 0  # Chunks unchanged, no migration needed
        self.skipped_existing_migration = 0  # Migration chunk already exists
        self.failed_chunks = 0
        self.embedded_chunks = 0  # New embeddings generated (for new chunks)
        self.reused_vectors = 0   # Old vectors reused (for changed chunks)
        # Classification-specific counters
        self.safe_reuse = 0           # SAFE_REUSE - reused old vector
        self.need_new_embedding = 0   # NEED_NEW_EMBEDDING - generated new embedding
        self.review = 0               # REVIEW - generated new embedding
        self.genuinely_new = 0        # NEW - generated new embedding
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


def _migration_chunk_exists(chunk_id: str, generation: int = 2) -> bool:
    """Check if a migration chunk (specific generation) already exists."""
    try:
        result = _rag_service.get_collection().get(ids=[chunk_id])
        if not result.get("ids") or len(result["ids"]) == 0:
            return False
        # Check if the chunk has the expected generation metadata
        # ChromaDB returns metadatas as list of lists: [[{...}]]
        metadatas = result.get("metadatas", [])
        if metadatas and len(metadatas) > 0 and len(metadatas[0]) > 0:
            meta = metadatas[0][0]
            return meta.get("generation", 1) == generation
        return False
    except Exception:
        return False


def _get_existing_vector(chunk_id: str) -> Optional[dict]:
    """Retrieve an existing vector from ChromaDB by chunk_id."""
    return _rag_service.get_vector_by_id(chunk_id)


def ingest_processed_document(
    processed_path: str,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_delay: float = _DEFAULT_BATCH_DELAY_SECONDS,
) -> IngestionStats:
    """Standard ingestion - preserves existing behavior exactly."""
    stats = IngestionStats()
    data = _load_processed_document(processed_path)

    if not data:
        stats.add_error(f"Failed to load processed document: {processed_path}")
        return stats

    chunks = data.get("chunks", [])
    stats.total_chunks = len(chunks)

    if not chunks:
        return stats

    quota_exhausted_globally = False

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
            stats.add_error(f"Missing chunk_id or text at index {idx} in {processed_path}")
            continue

        if _chunk_exists(chunk_id):
            stats.skipped_existing += 1
            continue

        # If quota was exhausted globally, skip remaining chunks and record them
        if quota_exhausted_globally:
            stats.add_error(
                f"Skipped chunk {chunk_id} due to earlier quota exhaustion",
                is_quota_exhausted=True,
            )
            continue

        try:
            embedding = generate_embedding(text)
            stats.embedded_chunks += 1
        except QuotaExhaustedError as exc:
            # Quota exhausted - fail fast, don't retry remaining chunks
            quota_exhausted_globally = True
            stats.add_error(
                f"Embedding quota exhausted for chunk {chunk_id}: {exc}",
                is_quota_exhausted=True,
            )
            continue
        except Exception as exc:
            stats.add_error(
                f"Embedding failed for chunk {chunk_id}: {exc}",
                is_quota_exhausted=False,
            )
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
            stats.add_error(
                f"Upsert failed for chunk {chunk_id}: {exc}",
                is_quota_exhausted=False,
            )

        if (idx + 1) % batch_size == 0 and idx + 1 < len(chunks):
            time.sleep(batch_delay)

    return stats


def ingest_migration_chunks(
    processed_path: str,
    comparison: Dict,  # Contains mapping from compare_processed_documents
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_delay: float = _DEFAULT_BATCH_DELAY_SECONDS,
    generation: int = 2,
) -> MigrationStats:
    """
    Controlled migration ingestion - only processes chunks that need migration.
    
    Does NOT automatically replace old vectors.
    Only creates new generation chunks for changed/new chunks.
    Preserves all existing generation 1 vectors.
    
    Uses explicit classifications from compare_processed_documents:
    - SAFE_REUSE: reuse superseded generation-1 vector
    - NEED_NEW_EMBEDDING: generate a new embedding
    - REVIEW: generate a new embedding
    - NEW: generate a new embedding
    """
    stats = MigrationStats()
    data = _load_processed_document(processed_path)

    if not data:
        stats.errors.append(f"Failed to load processed document: {processed_path}")
        return stats

    chunks = data.get("chunks", [])
    stats.total_chunks = len(chunks)

    if not chunks:
        return stats

    # Build lookup for new chunks by chunk_id
    new_chunks_by_id = {c["chunk_id"]: c for c in chunks}
    
    # Get classifications from comparison (new_chunk_id -> classification)
    classifications = comparison.get("classifications", {})
    
    # Get the list of chunks that need migration (changed + new)
    # comparison.mapping contains old_chunk_id -> new_chunk_id
    # comparison.new contains new_chunk_ids with no old counterpart
    # comparison.unchanged contains chunk_ids that are identical
    chunks_to_migrate = []
    
    for old_cid, new_cid in comparison.get("changed", []):
        if new_cid in new_chunks_by_id:
            chunks_to_migrate.append(("changed", new_cid, new_chunks_by_id[new_cid], old_cid))
    
    for new_cid in comparison.get("new", []):
        if new_cid in new_chunks_by_id:
            chunks_to_migrate.append(("new", new_cid, new_chunks_by_id[new_cid], None))

    # Track unchanged chunks
    stats.skipped_unchanged = len(comparison.get("unchanged", []))

    if not chunks_to_migrate:
        return stats

    quota_exhausted_globally = False

    for loop_idx, (change_type, chunk_id, chunk, supersedes_id) in enumerate(chunks_to_migrate):
        text = chunk.get("text", "")
        company_symbol = chunk.get("company_symbol", "")
        company_name = chunk.get("company_name", "")
        document_type = chunk.get("document_type", "")
        document_year = chunk.get("document_year")
        source = chunk.get("source", "")
        source_url = chunk.get("source_url", "")
        document_id = chunk.get("document_id", "")
        chunk_index = chunk.get("chunk_index", 0)
        section = chunk.get("section")

        if not chunk_id or not text:
            stats.errors.append(f"Missing chunk_id or text for {chunk_id} in {processed_path}")
            stats.failed_chunks += 1
            continue

        # Check if migration chunk already exists
        if _migration_chunk_exists(chunk_id, generation):
            stats.skipped_existing_migration += 1
            continue

        # If quota was exhausted globally, skip remaining chunks
        if quota_exhausted_globally:
            stats.errors.append(f"Skipped chunk {chunk_id} due to earlier quota exhaustion")
            stats.failed_chunks += 1
            continue

        # Determine classification for this chunk
        if change_type == "changed":
            classification = classifications.get(chunk_id, "REVIEW")
        else:  # new
            classification = "GENUINELY_NEW"

        embedding = None
        embedding_generated = False
        
        if classification == "SAFE_REUSE":
            # Reuse the old vector from the superseded chunk
            if supersedes_id:
                old_vector_data = _get_existing_vector(supersedes_id)
                if old_vector_data and old_vector_data.get("embedding"):
                    embedding = old_vector_data["embedding"]
                    stats.safe_reuse += 1
                    stats.reused_vectors += 1
                    stats.errors.append(
                        f"SAFE_REUSE: old_chunk_id={supersedes_id} -> new_chunk_id={chunk_id} "
                        f"(reused existing embedding, no Gemini call)"
                    )
                else:
                    # Old vector missing - skip safely, do NOT fall back to Gemini
                    stats.errors.append(
                        f"SAFE_REUSE MISSING: old_chunk_id={supersedes_id} -> new_chunk_id={chunk_id} "
                        f"(old vector not found in ChromaDB, chunk skipped for retry)"
                    )
                    stats.failed_chunks += 1
                    continue
            else:
                stats.errors.append(
                    f"SAFE_REUSE ERROR: new_chunk_id={chunk_id} has no supersedes_id"
                )
                stats.failed_chunks += 1
                continue
                
        elif classification in ("NEED_NEW_EMBEDDING", "REVIEW"):
            # Generate new embedding for these classifications
            try:
                embedding = generate_embedding(text)
                embedding_generated = True
                if classification == "NEED_NEW_EMBEDDING":
                    stats.need_new_embedding += 1
                else:  # REVIEW
                    stats.review += 1
                stats.errors.append(
                    f"{classification}: new_chunk_id={chunk_id} (generated via Gemini)"
                )
            except QuotaExhaustedError as exc:
                quota_exhausted_globally = True
                stats.errors.append(f"Embedding quota exhausted for chunk {chunk_id}: {exc}")
                stats.failed_chunks += 1
                continue
            except Exception as exc:
                stats.errors.append(f"Embedding failed for chunk {chunk_id}: {exc}")
                stats.failed_chunks += 1
                continue
                
        elif classification == "GENUINELY_NEW":
            # For genuinely new chunks: generate new embedding
            try:
                embedding = generate_embedding(text)
                embedding_generated = True
                stats.genuinely_new += 1
                stats.errors.append(f"GENUINELY_NEW: new_chunk_id={chunk_id} (generated via Gemini)")
            except QuotaExhaustedError as exc:
                quota_exhausted_globally = True
                stats.errors.append(f"Embedding quota exhausted for chunk {chunk_id}: {exc}")
                stats.failed_chunks += 1
                continue
            except Exception as exc:
                stats.errors.append(f"Embedding failed for chunk {chunk_id}: {exc}")
                stats.failed_chunks += 1
                continue

        # If we didn't get an embedding yet
        if embedding is None:
            stats.errors.append(f"No embedding available for chunk {chunk_id}")
            stats.failed_chunks += 1
            continue

        if embedding_generated:
            stats.embedded_chunks += 1

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

            # Use upsert with generation metadata and optional embedding
            _rag_service.upsert_document_chunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                generation=generation,
                supersedes_chunk_id=supersedes_id,
                embedding=embedding,
            )
            stats.migrated_chunks += 1
        except Exception as exc:
            stats.errors.append(f"Upsert failed for chunk {chunk_id}: {exc}")
            stats.failed_chunks += 1

        if (loop_idx + 1) % batch_size == 0 and loop_idx + 1 < len(chunks_to_migrate):
            time.sleep(batch_delay)

    return stats


def ingest_all_processed_documents(
    processed_dir: Optional[str] = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    batch_delay: float = _DEFAULT_BATCH_DELAY_SECONDS,
) -> IngestionStats:
    """Standard batch ingestion - preserves existing behavior exactly."""
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
            aggregate.quota_exhausted_chunks += stats.quota_exhausted_chunks
            aggregate.other_failed_chunks += stats.other_failed_chunks
            aggregate.errors.extend(stats.errors)
            aggregate.quota_exhausted_errors.extend(stats.quota_exhausted_errors)
            aggregate.other_errors.extend(stats.other_errors)

    return aggregate
