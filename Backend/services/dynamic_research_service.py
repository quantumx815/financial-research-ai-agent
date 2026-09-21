import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

from services.sec_service import resolve_ticker_to_cik, get_recent_filings, download_filing_content
from services.document_processor import process_document, DocumentMetadata
from services.bge_rag_service import bge_rag_service
from services.company_availability_service import company_availability_service


DEFAULT_FILING_LIMIT = 3
DEFAULT_FILING_TYPES = ["10-K", "10-Q"]


@dataclass
class FilingInfo:
    accession_number: str
    form: str
    filing_date: str
    primary_document: str
    document_url: str
    description: str


@dataclass
class DynamicResearchResult:
    symbol: str
    filings_discovered: List[FilingInfo] = field(default_factory=list)
    filings_downloaded: int = 0
    chunks_processed: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    skipped_duplicates: int = 0
    errors: List[str] = field(default_factory=list)


def _chunk_exists_in_bge(chunk_id: str) -> bool:
    """Check if a chunk already exists in the BGE collection."""
    try:
        result = bge_rag_service.get_collection().get(ids=[chunk_id])
        ids = result.get("ids", [])
        return bool(ids and ids[0] and len(ids[0]) > 0)
    except Exception:
        return False


def _ingest_chunks_to_bge(processed_doc, stats: DynamicResearchResult) -> None:
    """Ingest processed document chunks into BGE collection."""
    for chunk in processed_doc.chunks:
        chunk_id = chunk["chunk_id"]
        
        # Idempotency check
        if _chunk_exists_in_bge(chunk_id):
            stats.skipped_duplicates += 1
            continue
        
        metadata = DocumentMetadata(
            company_symbol=chunk.get("company_symbol", processed_doc.company_symbol),
            company_name=chunk.get("company_name", processed_doc.company_name),
            document_type=chunk.get("document_type", processed_doc.document_type),
            document_year=chunk.get("document_year", processed_doc.document_year),
            source=chunk.get("source", processed_doc.source),
            source_url=chunk.get("source_url", processed_doc.source_url),
            document_id=chunk.get("document_id", processed_doc.document_id),
            chunk_index=chunk.get("chunk_index", 0),
            section=chunk.get("section"),
        )
        
        try:
            bge_rag_service.add_document_chunk(
                chunk_id=chunk_id,
                text=chunk["text"],
                metadata=metadata,
            )
            stats.chunks_embedded += 1
            stats.chunks_stored += 1
        except Exception as exc:
            stats.errors.append(f"Failed to store chunk {chunk_id}: {exc}")


def research_and_ingest_company(
    symbol: str,
    filing_limit: int = DEFAULT_FILING_LIMIT,
    filing_types: List[str] = None,
) -> DynamicResearchResult:
    """
    Research SEC filings for a company and ingest into BGE collection.
    
    Args:
        symbol: Stock symbol (e.g., "MSFT")
        filing_limit: Maximum number of recent filings to process
        filing_types: List of filing types to fetch (default: 10-K, 10-Q)
        
    Returns:
        DynamicResearchResult with statistics
    """
    if filing_types is None:
        filing_types = DEFAULT_FILING_TYPES
    
    stats = DynamicResearchResult(symbol=symbol.upper())
    normalized_symbol = symbol.upper().strip()
    
    # Check if already available
    availability = company_availability_service.check_availability(normalized_symbol)
    if availability.available:
        stats.errors.append(f"Company {normalized_symbol} already available in BGE collection ({availability.chunk_count} chunks)")
        return stats
    
    # Step 1: Resolve ticker to CIK
    cik = resolve_ticker_to_cik(normalized_symbol)
    if not cik:
        stats.errors.append(f"Failed to resolve CIK for {normalized_symbol}")
        return stats
    
    # Step 2: Get recent filings
    try:
        filings = get_recent_filings(cik, filing_types, limit=filing_limit)
    except Exception as exc:
        stats.errors.append(f"SEC API error for {normalized_symbol}: {exc}")
        return stats
    
    if not filings:
        stats.errors.append(f"No recent {', '.join(filing_types)} filings found for {normalized_symbol}")
        return stats
    
    # Record discovered filings
    for filing in filings:
        stats.filings_discovered.append(FilingInfo(
            accession_number=filing["accession_number"],
            form=filing["form"],
            filing_date=filing["filing_date"],
            primary_document=filing["primary_document"],
            document_url=filing["document_url"],
            description=filing["description"],
        ))
    
    # Step 3: Process each filing
    for filing in filings:
        # Download filing content
        content = download_filing_content(filing["document_url"])
        if not content:
            stats.errors.append(f"Failed to download {filing['accession_number']} ({filing['form']})")
            continue
        
        stats.filings_downloaded += 1
        
        # Save to temp file for document processor
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".htm", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Prepare metadata
            doc_metadata = DocumentMetadata(
                company_symbol=normalized_symbol,
                company_name="",  # Will be filled by processor if available
                document_type=filing["form"],
                document_year=int(filing["filing_date"][:4]) if filing["filing_date"] else None,
                source="SEC",
                source_url=filing["document_url"],
                document_id=filing["accession_no_dashes"],
            )
            
            # Process document
            processed_doc = process_document(
                raw_path=tmp_path,
                metadata=doc_metadata,
                dry_run=False,
            )
            
            if not processed_doc:
                stats.errors.append(f"Failed to process {filing['accession_number']}")
                continue
            
            stats.chunks_processed += processed_doc.total_chunks
            
            # Ingest into BGE
            _ingest_chunks_to_bge(processed_doc, stats)
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
    return stats