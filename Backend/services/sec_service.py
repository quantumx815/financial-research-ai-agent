import os
from typing import Optional

import requests

from services.rag_service import DocumentMetadata


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{padded_cik}.json"
SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

SUPPORTED_FILING_TYPES = {"10-K", "10-Q"}


def _get_sec_headers() -> dict:
    app = os.getenv("SEC_USER_AGENT_APP", "FinancialResearchAI")
    email = os.getenv("SEC_USER_AGENT_EMAIL")

    if not email:
        raise EnvironmentError(
            "SEC_USER_AGENT_EMAIL is required. "
            "Set it in your environment to comply with SEC EDGAR access rules."
        )

    return {
        "User-Agent": f"{app} ({email})",
        "Accept": "application/json",
    }


def resolve_ticker_to_cik(ticker: str) -> Optional[str]:
    ticker = ticker.upper().strip()

    try:
        response = requests.get(
            SEC_TICKERS_URL,
            headers=_get_sec_headers(),
            timeout=15,
        )
        response.raise_for_status()
        tickers = response.json()
    except Exception:
        return None

    for item in tickers.values():
        if item.get("ticker", "").upper() == ticker:
            cik = str(item.get("cik_str", ""))
            return cik.zfill(10)

    return None


def get_latest_filing(cik: str, filing_types: list[str] | None = None):
    if filing_types is None:
        filing_types = list(SUPPORTED_FILING_TYPES)

    url = SEC_SUBMISSIONS_URL.format(padded_cik=cik)

    try:
        response = requests.get(
            url,
            headers=_get_sec_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    primary_documents = filings.get("primaryDocument", [])
    primary_document_descriptions = filings.get("primaryDocumentDescription", [])

    for index, form in enumerate(forms):
        if form not in filing_types:
            continue

        accession_raw = accession_numbers[index]
        accession_no_dashes = accession_raw.replace("-", "")
        filename = primary_documents[index]
        filing_date = filing_dates[index]
        description = primary_document_descriptions[index] if index < len(primary_document_descriptions) else form

        if not filename:
            continue

        document_url = (
            f"{SEC_ARCHIVE_BASE}/{int(cik)}/{accession_no_dashes}/{filename}"
        )

        return {
            "cik": cik,
            "accession_number": accession_raw,
            "accession_no_dashes": accession_no_dashes,
            "form": form,
            "filing_date": filing_date,
            "primary_document": filename,
            "description": description,
            "document_url": document_url,
        }

    return None


def download_filing_content(url: str) -> Optional[bytes]:
    try:
        response = requests.get(
            url,
            headers=_get_sec_headers(),
            timeout=30,
        )
        response.raise_for_status()
        return response.content
    except Exception:
        return None
