import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from services.rag_service import DocumentMetadata
from services.sec_service import (
    get_latest_filing,
    download_filing_content,
    resolve_ticker_to_cik,
)

_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data",
    "financial_documents",
    "raw",
)


@dataclass
class DocumentResult:
    success: bool
    status: str
    company_symbol: str
    company_name: Optional[str]
    document_type: Optional[str]
    filing_type: Optional[str]
    filing_date: Optional[str]
    document_year: Optional[int]
    source: str
    source_url: Optional[str]
    document_id: Optional[str]
    local_path: Optional[str]
    message: str


def _safe_filename(symbol: str, filing_type: str, accession_no_dashes: str) -> str:
    return f"{symbol}-{filing_type}-{accession_no_dashes}.html"


def _get_local_path(symbol: str, filename: str) -> str:
    symbol_dir = os.path.join(_RAW_DIR, symbol.upper())
    os.makedirs(symbol_dir, exist_ok=True)
    return os.path.join(symbol_dir, filename)


def _file_exists_and_valid(path: str, expected_size: int) -> bool:
    if not os.path.exists(path):
        return False
    if expected_size is not None and expected_size > 0:
        return os.path.getsize(path) == expected_size
    return os.path.getsize(path) > 0


def discover_document(symbol: str):
    cik = resolve_ticker_to_cik(symbol)
    if not cik:
        return None, "company_not_found_in_sec"

    filing = get_latest_filing(cik)
    if not filing:
        return None, "no_suitable_filing_found"

    return filing, "discovered"


def download_document(symbol: str, filing_info: dict) -> DocumentResult:
    symbol = symbol.upper()
    filename = _safe_filename(symbol, filing_info["form"], filing_info["accession_no_dashes"])
    local_path = _get_local_path(symbol, filename)

    content = download_filing_content(filing_info["document_url"])

    if content is None:
        return DocumentResult(
            success=False,
            status="download_failed",
            company_symbol=symbol,
            company_name=None,
            document_type=filing_info["form"],
            filing_type=filing_info["form"],
            filing_date=filing_info["filing_date"],
            document_year=int(filing_info["filing_date"][:4]) if filing_info["filing_date"] else None,
            source="SEC",
            source_url=filing_info["document_url"],
            document_id=filing_info["accession_number"],
            local_path=None,
            message="Document download failed. SEC may be temporarily unavailable.",
        )

    if not content:
        return DocumentResult(
            success=False,
            status="invalid_document",
            company_symbol=symbol,
            company_name=None,
            document_type=filing_info["form"],
            filing_type=filing_info["form"],
            filing_date=filing_info["filing_date"],
            document_year=int(filing_info["filing_date"][:4]) if filing_info["filing_date"] else None,
            source="SEC",
            source_url=filing_info["document_url"],
            document_id=filing_info["accession_number"],
            local_path=None,
            message="Downloaded document was empty.",
        )

    try:
        with open(local_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        return DocumentResult(
            success=False,
            status="storage_failed",
            company_symbol=symbol,
            company_name=None,
            document_type=filing_info["form"],
            filing_type=filing_info["form"],
            filing_date=filing_info["filing_date"],
            document_year=int(filing_info["filing_date"][:4]) if filing_info["filing_date"] else None,
            source="SEC",
            source_url=filing_info["document_url"],
            document_id=filing_info["accession_number"],
            local_path=None,
            message=f"Failed to save document: {exc}",
        )

    return DocumentResult(
        success=True,
        status="downloaded",
        company_symbol=symbol,
        company_name=None,
        document_type=filing_info["form"],
        filing_type=filing_info["form"],
        filing_date=filing_info["filing_date"],
        document_year=int(filing_info["filing_date"][:4]) if filing_info["filing_date"] else None,
        source="SEC",
        source_url=filing_info["document_url"],
        document_id=filing_info["accession_number"],
        local_path=local_path,
        message="Document downloaded successfully.",
    )


def ensure_document(symbol: str) -> DocumentResult:
    symbol = symbol.upper()

    filing_info, discovery_status = discover_document(symbol)

    if discovery_status == "company_not_found_in_sec":
        return DocumentResult(
            success=False,
            status="company_not_found",
            company_symbol=symbol,
            company_name=None,
            document_type=None,
            filing_type=None,
            filing_date=None,
            document_year=None,
            source="SEC",
            source_url=None,
            document_id=None,
            local_path=None,
            message="Company symbol was not found in SEC EDGAR records.",
        )

    if discovery_status == "no_suitable_filing_found":
        return DocumentResult(
            success=False,
            status="document_unavailable",
            company_symbol=symbol,
            company_name=None,
            document_type=None,
            filing_type=None,
            filing_date=None,
            document_year=None,
            source="SEC",
            source_url=None,
            document_id=None,
            local_path=None,
            message="No suitable trusted filing was available.",
        )

    filename = _safe_filename(symbol, filing_info["form"], filing_info["accession_no_dashes"])
    local_path = _get_local_path(symbol, filename)

    if _file_exists_and_valid(local_path, expected_size=None):
        return DocumentResult(
            success=True,
            status="cached",
            company_symbol=symbol,
            company_name=None,
            document_type=filing_info["form"],
            filing_type=filing_info["form"],
            filing_date=filing_info["filing_date"],
            document_year=int(filing_info["filing_date"][:4]) if filing_info["filing_date"] else None,
            source="SEC",
            source_url=filing_info["document_url"],
            document_id=filing_info["accession_number"],
            local_path=local_path,
            message="Existing document reused.",
        )

    return download_document(symbol, filing_info)


def get_document_metadata(result: DocumentResult) -> Optional[DocumentMetadata]:
    if not result.success:
        return None

    return DocumentMetadata(
        company_symbol=result.company_symbol,
        company_name=result.company_name or "",
        document_type=result.document_type or "",
        document_year=result.document_year,
        source=result.source,
        source_url=result.source_url,
        document_id=result.document_id or "",
    )
