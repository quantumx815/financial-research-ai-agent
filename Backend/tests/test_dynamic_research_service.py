import os
import sys
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.dynamic_research_service import (
    research_and_ingest_company,
    DynamicResearchResult,
    FilingInfo,
    _chunk_exists_in_bge,
    _ingest_chunks_to_bge,
)


class TestDynamicResearchService:
    """Unit tests for dynamic research service - all external deps mocked."""

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    @patch("services.dynamic_research_service.download_filing_content")
    @patch("services.dynamic_research_service.process_document")
    @patch("services.dynamic_research_service.company_availability_service")
    @patch("services.dynamic_research_service.bge_rag_service")
    def test_successful_dynamic_research_flow(
        self,
        mock_bge_service,
        mock_avail_service,
        mock_process,
        mock_download,
        mock_get_filings,
        mock_resolve_cik,
    ):
        # Mock: company not available
        mock_avail = MagicMock()
        mock_avail.available = False
        mock_avail.chunk_count = 0
        mock_avail_service.check_availability.return_value = mock_avail

        # Mock: SEC CIK resolution
        mock_resolve_cik.return_value = "0000123456"

        # Mock: SEC filings discovery
        mock_get_filings.return_value = [
            {
                "cik": "0000123456",
                "accession_number": "0001234567-26-000001",
                "accession_no_dashes": "000123456726000001",
                "form": "10-Q",
                "filing_date": "2026-06-30",
                "primary_document": "test-10q.htm",
                "description": "Quarterly Report",
                "document_url": "https://www.sec.gov/Archives/edgar/data/123456/000123456726000001/test-10q.htm",
            },
            {
                "cik": "0000123456",
                "accession_number": "0001234567-25-000002",
                "accession_no_dashes": "000123456725000002",
                "form": "10-K",
                "filing_date": "2025-12-31",
                "primary_document": "test-10k.htm",
                "description": "Annual Report",
                "document_url": "https://www.sec.gov/Archives/edgar/data/123456/000123456725000002/test-10k.htm",
            },
        ]

        # Mock: SEC download
        mock_download.return_value = b"<html><body>Test filing content</body></html>"

        # Mock: document processing
        mock_processed = MagicMock()
        mock_processed.total_chunks = 3
        mock_processed.company_symbol = "TESTDYN"
        mock_processed.company_name = "Test Dynamic Company"
        mock_processed.document_type = "10-Q"
        mock_processed.document_year = 2026
        mock_processed.source = "SEC"
        mock_processed.source_url = "https://example.com"
        mock_processed.document_id = "000123456726000001"
        mock_processed.chunks = [
            {
                "chunk_id": "chunk-1",
                "text": "Test chunk 1",
                "company_symbol": "TESTDYN",
                "company_name": "Test Dynamic Company",
                "document_type": "10-Q",
                "document_year": 2026,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "000123456726000001",
                "chunk_index": 0,
                "section": "Item 1A - Risk Factors",
            },
            {
                "chunk_id": "chunk-2",
                "text": "Test chunk 2",
                "company_symbol": "TESTDYN",
                "company_name": "Test Dynamic Company",
                "document_type": "10-Q",
                "document_year": 2026,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "000123456726000001",
                "chunk_index": 1,
                "section": "Item 7 - MD&A",
            },
            {
                "chunk_id": "chunk-3",
                "text": "Test chunk 3",
                "company_symbol": "TESTDYN",
                "company_name": "Test Dynamic Company",
                "document_type": "10-Q",
                "document_year": 2026,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "000123456726000001",
                "chunk_index": 2,
                "section": "Item 8 - Financial Statements",
            },
        ]
        mock_process.return_value = mock_processed

        # Mock: BGE chunk existence check (not existing)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [[]]}
        mock_bge_service.get_collection.return_value = mock_collection

        # Mock: BGE add_document_chunk (verify it's called with correct data)
        mock_bge_service.add_document_chunk = MagicMock()

        # Run
        result = research_and_ingest_company("TESTDYN", filing_limit=2)

        # Assert: result structure
        assert isinstance(result, DynamicResearchResult)
        assert result.symbol == "TESTDYN"
        assert len(result.filings_discovered) == 2
        assert result.filings_downloaded == 2
        assert result.chunks_processed == 6  # 3 chunks * 2 filings
        assert result.chunks_embedded == 6
        assert result.chunks_stored == 6
        assert result.skipped_duplicates == 0
        assert len(result.errors) == 0

        # Assert: BGE add_document_chunk called 6 times with correct data
        assert mock_bge_service.add_document_chunk.call_count == 6
        for call in mock_bge_service.add_document_chunk.call_args_list:
            args, kwargs = call
            assert "chunk_id" in kwargs
            assert "text" in kwargs
            assert "metadata" in kwargs
            meta = kwargs["metadata"]
            assert meta.company_symbol == "TESTDYN"
            assert meta.document_id == "000123456726000001"

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    def test_sec_discovery_failure_no_cik(self, mock_resolve_cik):
        mock_resolve_cik.return_value = None

        result = research_and_ingest_company("UNKNOWN")

        assert result.symbol == "UNKNOWN"
        assert len(result.errors) == 1
        assert "Failed to resolve CIK" in result.errors[0]

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    def test_sec_discovery_failure_no_filings(self, mock_get_filings, mock_resolve_cik):
        mock_resolve_cik.return_value = "0000123456"
        mock_get_filings.return_value = []

        result = research_and_ingest_company("TESTDYN")

        assert result.symbol == "TESTDYN"
        assert len(result.errors) == 1
        assert "No recent" in result.errors[0]

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    def test_sec_api_exception(self, mock_get_filings, mock_resolve_cik):
        mock_resolve_cik.return_value = "0000123456"
        mock_get_filings.side_effect = Exception("SEC API timeout")

        result = research_and_ingest_company("TESTDYN")

        assert result.symbol == "TESTDYN"
        assert len(result.errors) == 1
        assert "SEC API error" in result.errors[0]

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    @patch("services.dynamic_research_service.download_filing_content")
    @patch("services.dynamic_research_service.company_availability_service")
    def test_download_failure(
        self, mock_avail_service, mock_download, mock_get_filings, mock_resolve_cik
    ):
        mock_avail = MagicMock()
        mock_avail.available = False
        mock_avail.chunk_count = 0
        mock_avail_service.check_availability.return_value = mock_avail

        mock_resolve_cik.return_value = "0000123456"
        mock_get_filings.return_value = [
            {
                "cik": "0000123456",
                "accession_number": "0001234567-26-000001",
                "accession_no_dashes": "000123456726000001",
                "form": "10-Q",
                "filing_date": "2026-06-30",
                "primary_document": "test-10q.htm",
                "description": "Quarterly Report",
                "document_url": "https://example.com/test.htm",
            }
        ]
        mock_download.return_value = None  # Download failed

        result = research_and_ingest_company("TESTDYN")

        assert result.filings_downloaded == 0
        assert len(result.errors) == 1
        assert "Failed to download" in result.errors[0]

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    @patch("services.dynamic_research_service.download_filing_content")
    @patch("services.dynamic_research_service.process_document")
    @patch("services.dynamic_research_service.company_availability_service")
    def test_processing_failure(
        self, mock_avail_service, mock_process, mock_download, mock_get_filings, mock_resolve_cik
    ):
        mock_avail = MagicMock()
        mock_avail.available = False
        mock_avail.chunk_count = 0
        mock_avail_service.check_availability.return_value = mock_avail

        mock_resolve_cik.return_value = "0000123456"
        mock_get_filings.return_value = [
            {
                "cik": "0000123456",
                "accession_number": "0001234567-26-000001",
                "accession_no_dashes": "000123456726000001",
                "form": "10-Q",
                "filing_date": "2026-06-30",
                "primary_document": "test-10q.htm",
                "description": "Quarterly Report",
                "document_url": "https://example.com/test.htm",
            }
        ]
        mock_download.return_value = b"<html>Test</html>"
        mock_process.return_value = None  # Processing failed

        result = research_and_ingest_company("TESTDYN")

        assert result.chunks_processed == 0
        assert len(result.errors) == 1
        assert "Failed to process" in result.errors[0]

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    @patch("services.dynamic_research_service.download_filing_content")
    @patch("services.dynamic_research_service.process_document")
    @patch("services.dynamic_research_service.company_availability_service")
    @patch("services.dynamic_research_service.bge_rag_service")
    def test_idempotent_ingestion_no_duplicates(
        self,
        mock_bge_service,
        mock_avail_service,
        mock_process,
        mock_download,
        mock_get_filings,
        mock_resolve_cik,
    ):
        # Mock: company not available (bypasses early exit)
        mock_avail = MagicMock()
        mock_avail.available = False
        mock_avail.chunk_count = 0
        mock_avail_service.check_availability.return_value = mock_avail

        mock_resolve_cik.return_value = "0000123456"
        mock_get_filings.return_value = [
            {
                "cik": "0000123456",
                "accession_number": "0001234567-26-000001",
                "accession_no_dashes": "000123456726000001",
                "form": "10-Q",
                "filing_date": "2026-06-30",
                "primary_document": "test-10q.htm",
                "description": "Quarterly Report",
                "document_url": "https://example.com/test.htm",
            }
        ]
        mock_download.return_value = b"<html>Test</html>"

        mock_processed = MagicMock()
        mock_processed.total_chunks = 2
        mock_processed.company_symbol = "TESTDYN2"
        mock_processed.company_name = "Test Dynamic Company 2"
        mock_processed.document_type = "10-Q"
        mock_processed.document_year = 2026
        mock_processed.source = "SEC"
        mock_processed.source_url = "https://example.com"
        mock_processed.document_id = "000123456726000001"
        mock_processed.chunks = [
            {
                "chunk_id": "existing-chunk-1",
                "text": "Test chunk 1",
                "company_symbol": "TESTDYN2",
                "company_name": "Test Dynamic Company 2",
                "document_type": "10-Q",
                "document_year": 2026,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "000123456726000001",
                "chunk_index": 0,
                "section": "Item 1A - Risk Factors",
            },
            {
                "chunk_id": "existing-chunk-2",
                "text": "Test chunk 2",
                "company_symbol": "TESTDYN2",
                "company_name": "Test Dynamic Company 2",
                "document_type": "10-Q",
                "document_year": 2026,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "000123456726000001",
                "chunk_index": 1,
                "section": "Item 7 - MD&A",
            },
        ]
        mock_process.return_value = mock_processed

        # Mock: BGE collection - first call returns empty (not exists), second returns existing
        call_count = {"count": 0}

        def mock_collection_get(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] <= 2:  # First run: 2 chunks checked, not found
                return {"ids": [[]]}
            else:  # Second run: chunks exist
                return {"ids": [["existing-chunk-1", "existing-chunk-2"]]}

        mock_collection = MagicMock()
        mock_collection.get.side_effect = mock_collection_get
        mock_bge_service.get_collection.return_value = mock_collection
        mock_bge_service.add_document_chunk = MagicMock()

        # First run - should store chunks
        result1 = research_and_ingest_company("TESTDYN2")
        assert result1.chunks_stored == 2
        assert result1.skipped_duplicates == 0
        assert mock_bge_service.add_document_chunk.call_count == 2

        # Reset mock for second run verification
        mock_bge_service.add_document_chunk.reset_mock()

        # Second run - should skip duplicates at chunk level
        result2 = research_and_ingest_company("TESTDYN2")
        assert result2.chunks_stored == 0
        assert result2.skipped_duplicates == 2
        assert mock_bge_service.add_document_chunk.call_count == 0

    @patch("services.dynamic_research_service.resolve_ticker_to_cik")
    @patch("services.dynamic_research_service.get_recent_filings")
    @patch("services.dynamic_research_service.download_filing_content")
    @patch("services.dynamic_research_service.process_document")
    @patch("services.dynamic_research_service.company_availability_service")
    @patch("services.dynamic_research_service.bge_rag_service")
    def test_embedding_storage_failure(
        self,
        mock_bge_service,
        mock_avail_service,
        mock_process,
        mock_download,
        mock_get_filings,
        mock_resolve_cik,
    ):
        """Test that embedding/storage failures are captured in errors."""
        mock_avail = MagicMock()
        mock_avail.available = False
        mock_avail.chunk_count = 0
        mock_avail_service.check_availability.return_value = mock_avail

        mock_resolve_cik.return_value = "0000123456"
        mock_get_filings.return_value = [
            {
                "cik": "0000123456",
                "accession_number": "0001234567-26-000001",
                "accession_no_dashes": "000123456726000001",
                "form": "10-Q",
                "filing_date": "2026-06-30",
                "primary_document": "test-10q.htm",
                "description": "Quarterly Report",
                "document_url": "https://example.com/test.htm",
            }
        ]
        mock_download.return_value = b"<html>Test</html>"

        mock_processed = MagicMock()
        mock_processed.total_chunks = 1
        mock_processed.company_symbol = "TESTDYN3"
        mock_processed.company_name = "Test Company 3"
        mock_processed.document_type = "10-Q"
        mock_processed.document_year = 2026
        mock_processed.source = "SEC"
        mock_processed.source_url = "https://example.com"
        mock_processed.document_id = "000123456726000001"
        mock_processed.chunks = [
            {
                "chunk_id": "chunk-fail-1",
                "text": "Test chunk",
                "company_symbol": "TESTDYN3",
                "company_name": "Test Company 3",
                "document_type": "10-Q",
                "document_year": 2026,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "000123456726000001",
                "chunk_index": 0,
                "section": "Item 1A - Risk Factors",
            },
        ]
        mock_process.return_value = mock_processed

        # Mock: BGE collection - chunk doesn't exist
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [[]]}
        mock_bge_service.get_collection.return_value = mock_collection

        # Mock: add_document_chunk raises exception
        mock_bge_service.add_document_chunk.side_effect = Exception("ChromaDB write failed")

        result = research_and_ingest_company("TESTDYN3")

        assert result.chunks_stored == 0
        assert len(result.errors) == 1
        assert "Failed to store chunk" in result.errors[0]
        assert "ChromaDB write failed" in result.errors[0]

    @patch("services.dynamic_research_service.bge_rag_service")
    def test_chunk_exists_in_bge_returns_true(self, mock_bge_service):
        """Test _chunk_exists_in_bge returns True when chunk found."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [["existing-chunk-1"]]}
        mock_bge_service.get_collection.return_value = mock_collection

        result = _chunk_exists_in_bge("existing-chunk-1")
        assert result is True

    @patch("services.dynamic_research_service.bge_rag_service")
    def test_chunk_exists_in_bge_returns_false(self, mock_bge_service):
        """Test _chunk_exists_in_bge returns False when chunk not found."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [[]]}
        mock_bge_service.get_collection.return_value = mock_collection

        result = _chunk_exists_in_bge("non-existent-chunk")
        assert result is False

    @patch("services.dynamic_research_service.bge_rag_service")
    def test_chunk_exists_in_bge_handles_exception(self, mock_bge_service):
        """Test _chunk_exists_in_bge returns False on exception."""
        mock_bge_service.get_collection.side_effect = Exception("ChromaDB error")

        result = _chunk_exists_in_bge("any-chunk")
        assert result is False

    def test_result_statistics_structure(self):
        """Test DynamicResearchResult dataclass defaults."""
        result = DynamicResearchResult(symbol="TEST")
        assert result.symbol == "TEST"
        assert result.filings_discovered == []
        assert result.filings_downloaded == 0
        assert result.chunks_processed == 0
        assert result.chunks_embedded == 0
        assert result.chunks_stored == 0
        assert result.skipped_duplicates == 0
        assert result.errors == []

    def test_filing_info_dataclass(self):
        """Test FilingInfo dataclass construction."""
        filing = FilingInfo(
            accession_number="0001234567-26-000001",
            form="10-Q",
            filing_date="2026-06-30",
            primary_document="test.htm",
            document_url="https://example.com/test.htm",
            description="Quarterly Report",
        )
        assert filing.accession_number == "0001234567-26-000001"
        assert filing.form == "10-Q"
        assert filing.filing_date == "2026-06-30"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])