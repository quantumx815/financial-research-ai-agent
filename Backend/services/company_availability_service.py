from dataclasses import dataclass
from typing import Optional

from services.bge_rag_service import bge_rag_service


@dataclass
class CompanyAvailability:
    symbol: str
    available: bool
    chunk_count: int
    document_count: int
    fresh_research_required: bool
    message: Optional[str] = None


class CompanyAvailabilityService:
    def __init__(self):
        self._service = bge_rag_service

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        if not symbol:
            return ""
        return symbol.strip().upper()

    def check_availability(self, symbol: str) -> CompanyAvailability:
        normalized = self.normalize_symbol(symbol)

        if not normalized:
            return CompanyAvailability(
                symbol=normalized,
                available=False,
                chunk_count=0,
                document_count=0,
                fresh_research_required=True,
                message="Invalid or empty symbol",
            )

        try:
            results = self._service.get_collection().get(
                where={"company_symbol": normalized},
                include=["metadatas"],
            )

            chunk_ids = results.get("ids", [])
            chunk_count = len(chunk_ids)

            if chunk_count == 0:
                return CompanyAvailability(
                    symbol=normalized,
                    available=False,
                    chunk_count=0,
                    document_count=0,
                    fresh_research_required=True,
                    message=f"No indexed documents found for {normalized}",
                )

            metadatas = results.get("metadatas", [])
            document_ids = set()
            for meta in metadatas:
                if meta and "document_id" in meta:
                    document_ids.add(meta["document_id"])

            document_count = len(document_ids)

            return CompanyAvailability(
                symbol=normalized,
                available=True,
                chunk_count=chunk_count,
                document_count=document_count,
                fresh_research_required=False,
                message=f"{normalized} available with {chunk_count} chunks across {document_count} document(s)",
            )

        except Exception as exc:
            return CompanyAvailability(
                symbol=normalized,
                available=False,
                chunk_count=0,
                document_count=0,
                fresh_research_required=True,
                message=f"Error checking availability: {exc}",
            )


company_availability_service = CompanyAvailabilityService()