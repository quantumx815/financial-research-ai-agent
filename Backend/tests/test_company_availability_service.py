import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.bge_rag_service import BGERAGService, DocumentMetadata
from services.company_availability_service import CompanyAvailabilityService, CompanyAvailability


class TestCompanyAvailabilityService:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.service = CompanyAvailabilityService()
        self.bge_service = BGERAGService()
        # Clean up test companies before each test
        for sym in ["TESTAVAIL", "TESTAVAIL2", "TESTNORM"]:
            self.bge_service.delete_company_documents(sym)
        yield
        # Clean up after
        for sym in ["TESTAVAIL", "TESTAVAIL2", "TESTNORM"]:
            self.bge_service.delete_company_documents(sym)

    def test_existing_company_available(self):
        # Add test data for TESTAVAIL
        metadata = DocumentMetadata(
            company_symbol="TESTAVAIL",
            company_name="Test Available Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-avail-doc-001",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )
        self.bge_service.add_document_chunk(
            chunk_id="test-avail-chunk-1",
            text="Test chunk for available company.",
            metadata=metadata,
        )
        self.bge_service.add_document_chunk(
            chunk_id="test-avail-chunk-2",
            text="Second chunk for available company.",
            metadata=metadata,
        )

        result = self.service.check_availability("TESTAVAIL")

        assert result.symbol == "TESTAVAIL"
        assert result.available is True
        assert result.chunk_count == 2
        assert result.document_count == 1
        assert result.fresh_research_required is False
        assert "available" in result.message.lower()

    def test_unknown_company_not_available(self):
        result = self.service.check_availability("UNKNOWNXYZ")

        assert result.symbol == "UNKNOWNXYZ"
        assert result.available is False
        assert result.chunk_count == 0
        assert result.document_count == 0
        assert result.fresh_research_required is True
        assert "no indexed documents" in result.message.lower()

    def test_symbol_normalization_lowercase(self):
        metadata = DocumentMetadata(
            company_symbol="TESTNORM",
            company_name="Test Norm Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-norm-doc-001",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )
        self.bge_service.add_document_chunk(
            chunk_id="test-norm-chunk-1",
            text="Test chunk for normalization.",
            metadata=metadata,
        )

        result = self.service.check_availability("testnorm")

        assert result.symbol == "TESTNORM"
        assert result.available is True
        assert result.chunk_count == 1

    def test_symbol_normalization_mixed_case(self):
        result = self.service.check_availability("TeStNoRm")

        assert result.symbol == "TESTNORM"

    def test_symbol_normalization_whitespace(self):
        result = self.service.check_availability("  testnorm  ")

        assert result.symbol == "TESTNORM"

    def test_empty_symbol_handling(self):
        result = self.service.check_availability("")

        assert result.symbol == ""
        assert result.available is False
        assert result.chunk_count == 0
        assert result.document_count == 0
        assert result.fresh_research_required is True
        assert "invalid or empty" in result.message.lower()

    def test_none_symbol_handling(self):
        result = self.service.check_availability(None)

        assert result.symbol == ""
        assert result.available is False
        assert result.fresh_research_required is True

    def test_whitespace_only_symbol_handling(self):
        result = self.service.check_availability("   ")

        assert result.symbol == ""
        assert result.available is False
        assert result.fresh_research_required is True

    def test_correct_indexed_count_multiple_documents(self):
        # Add chunks from two different documents
        metadata1 = DocumentMetadata(
            company_symbol="TESTAVAIL2",
            company_name="Test Multi Doc Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-multi-doc-001",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )
        metadata2 = DocumentMetadata(
            company_symbol="TESTAVAIL2",
            company_name="Test Multi Doc Company",
            document_type="10-K",
            document_year=2025,
            source="SEC",
            source_url="https://example.com",
            document_id="test-multi-doc-002",
            chunk_index=0,
            section="Item 7 - MD&A",
        )

        # 3 chunks from first document
        for i in range(3):
            self.bge_service.add_document_chunk(
                chunk_id=f"test-multi-chunk-{i}",
                text=f"Test chunk {i} from first document.",
                metadata=metadata1,
            )

        # 2 chunks from second document
        for i in range(2):
            self.bge_service.add_document_chunk(
                chunk_id=f"test-multi-chunk-{i+10}",
                text=f"Test chunk {i} from second document.",
                metadata=metadata2,
            )

        result = self.service.check_availability("TESTAVAIL2")

        assert result.available is True
        assert result.chunk_count == 5
        assert result.document_count == 2

    def test_normalize_symbol_static_method(self):
        assert CompanyAvailabilityService.normalize_symbol("aapl") == "AAPL"
        assert CompanyAvailabilityService.normalize_symbol("  TSLA  ") == "TSLA"
        assert CompanyAvailabilityService.normalize_symbol("NvDa") == "NVDA"
        assert CompanyAvailabilityService.normalize_symbol("") == ""
        assert CompanyAvailabilityService.normalize_symbol("   ") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])