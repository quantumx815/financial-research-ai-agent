import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.bge_query_service import BGEQueryService, BGEQueryResponse
from services.bge_rag_service import RetrievalResult


class TestBGEQueryService:
    """Unit tests for BGE query service - all external deps mocked."""

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_successful_retrieval(
        self, mock_avail_service, mock_bge_service
    ):
        """Test successful retrieval for available company."""
        # Mock: company available
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        # Mock: BGE query returns results
        mock_results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Test chunk about financial risks",
                company_symbol="TEST",
                document_type="10-Q",
                document_year=2026,
                source="SEC",
                source_url="https://example.com",
                document_id="doc-001",
                chunk_index=0,
                distance=0.35,
                embedding_model="BAAI/bge-small-en-v1.5",
                section="Item 1A - Risk Factors",
            ),
            RetrievalResult(
                chunk_id="chunk-2",
                chunk_text="Another chunk about revenue",
                company_symbol="TEST",
                document_type="10-Q",
                document_year=2026,
                source="SEC",
                source_url="https://example.com",
                document_id="doc-001",
                chunk_index=1,
                distance=0.42,
                embedding_model="BAAI/bge-small-en-v1.5",
                section="Item 7 - MD&A",
            ),
        ]
        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = mock_results
        mock_retrieval_response.total_found = 2
        mock_retrieval_response.error = None
        mock_bge_service.query.return_value = mock_retrieval_response

        # Run
        service = BGEQueryService()
        response = service.query("TEST", "What are the financial risks?", top_k=5)

        # Assert
        assert isinstance(response, BGEQueryResponse)
        assert response.symbol == "TEST"
        assert response.query == "What are the financial risks?"
        assert response.top_k == 5
        assert response.available is True
        assert response.error is None
        assert len(response.results) == 2
        assert response.total_found == 2
        assert response.results[0].chunk_id == "chunk-1"
        assert response.results[0].section == "Item 1A - Risk Factors"
        assert response.results[0].distance == 0.35

        # Verify mocks called correctly
        mock_avail_service.check_availability.assert_called_once_with("TEST")
        mock_bge_service.query.assert_called_once_with(
            query_text="What are the financial risks?",
            company_symbol="TEST",
            n_results=5,
        )

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_company_filtering(
        self, mock_avail_service, mock_bge_service
    ):
        """Test that company filtering is applied."""
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = []
        mock_retrieval_response.total_found = 0
        mock_retrieval_response.error = None
        mock_bge_service.query.return_value = mock_retrieval_response

        service = BGEQueryService()
        response = service.query("TEST", "test query", top_k=3)

        # Verify BGE query called with correct company_symbol
        mock_bge_service.query.assert_called_once()
        call_args = mock_bge_service.query.call_args
        assert call_args.kwargs["company_symbol"] == "TEST"
        assert call_args.kwargs["n_results"] == 3

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_empty_query_handled(
        self, mock_avail_service, mock_bge_service
    ):
        """Test empty query returns error."""
        service = BGEQueryService()
        response = service.query("TEST", "", top_k=5)

        assert response.available is False
        assert response.error == "Query text is empty."
        assert response.symbol == "TEST"
        # BGE service should not be called
        mock_bge_service.query.assert_not_called()

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_empty_symbol_handled(
        self, mock_avail_service, mock_bge_service
    ):
        """Test empty symbol returns error."""
        service = BGEQueryService()
        response = service.query("", "test query", top_k=5)

        assert response.available is False
        assert response.error == "Invalid or empty symbol"
        assert response.symbol == ""
        mock_bge_service.query.assert_not_called()

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_company_not_available(
        self, mock_avail_service, mock_bge_service
    ):
        """Test company not in knowledge base."""
        mock_avail = MagicMock()
        mock_avail.available = False
        mock_avail.chunk_count = 0
        mock_avail_service.check_availability.return_value = mock_avail

        service = BGEQueryService()
        response = service.query("UNKNOWN", "test query", top_k=5)

        assert response.available is False
        assert "not indexed" in response.error
        assert response.symbol == "UNKNOWN"
        mock_bge_service.query.assert_not_called()

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_no_results_found(
        self, mock_avail_service, mock_bge_service
    ):
        """Test query with no matching results."""
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = []
        mock_retrieval_response.total_found = 0
        mock_retrieval_response.error = None
        mock_bge_service.query.return_value = mock_retrieval_response

        service = BGEQueryService()
        response = service.query("TEST", "nonexistent topic", top_k=5)

        assert response.available is True
        assert response.total_found == 0
        assert len(response.results) == 0
        assert response.error is None

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_chroma_failure_handled(
        self, mock_avail_service, mock_bge_service
    ):
        """Test ChromaDB failure is captured in response."""
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = []
        mock_retrieval_response.total_found = 0
        mock_retrieval_response.error = "ChromaDB query failed: connection timeout"
        mock_bge_service.query.return_value = mock_retrieval_response

        service = BGEQueryService()
        response = service.query("TEST", "test query", top_k=5)

        assert response.available is True
        assert response.error == "ChromaDB query failed: connection timeout"
        assert len(response.results) == 0

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_embedding_failure_handled(
        self, mock_avail_service, mock_bge_service
    ):
        """Test embedding generation failure is captured."""
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = []
        mock_retrieval_response.total_found = 0
        mock_retrieval_response.error = "Embedding generation failed: model load error"
        mock_bge_service.query.return_value = mock_retrieval_response

        service = BGEQueryService()
        response = service.query("TEST", "test query", top_k=5)

        assert response.available is True
        assert "Embedding generation failed" in response.error

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_top_k_respected(
        self, mock_avail_service, mock_bge_service
    ):
        """Test top_k parameter is passed through."""
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = []
        mock_retrieval_response.total_found = 0
        mock_retrieval_response.error = None
        mock_bge_service.query.return_value = mock_retrieval_response

        service = BGEQueryService()
        response = service.query("TEST", "test query", top_k=3)

        assert response.top_k == 3
        call_kwargs = mock_bge_service.query.call_args.kwargs
        assert call_kwargs["n_results"] == 3

    @patch("services.bge_query_service.bge_rag_service")
    @patch("services.bge_query_service.company_availability_service")
    def test_result_structure_complete(
        self, mock_avail_service, mock_bge_service
    ):
        """Test response contains all required fields."""
        mock_avail = MagicMock()
        mock_avail.available = True
        mock_avail.chunk_count = 10
        mock_avail_service.check_availability.return_value = mock_avail

        mock_result = RetrievalResult(
            chunk_id="chunk-1",
            chunk_text="Full chunk text content",
            company_symbol="TEST",
            document_type="10-K",
            document_year=2025,
            source="SEC",
            source_url="https://sec.gov/...",
            document_id="0001234567-25-000001",
            chunk_index=5,
            distance=0.28,
            embedding_model="BAAI/bge-small-en-v1.5",
            section="Item 1A - Risk Factors",
        )
        mock_retrieval_response = MagicMock()
        mock_retrieval_response.results = [mock_result]
        mock_retrieval_response.total_found = 1
        mock_retrieval_response.error = None
        mock_bge_service.query.return_value = mock_retrieval_response

        service = BGEQueryService()
        response = service.query("TEST", "risk factors", top_k=5)

        assert response.available is True
        assert len(response.results) == 1
        result = response.results[0]
        # Verify all required fields present
        assert result.chunk_id == "chunk-1"
        assert result.chunk_text == "Full chunk text content"
        assert result.company_symbol == "TEST"
        assert result.document_type == "10-K"
        assert result.document_year == 2025
        assert result.source == "SEC"
        assert result.source_url == "https://sec.gov/..."
        assert result.document_id == "0001234567-25-000001"
        assert result.chunk_index == 5
        assert result.distance == 0.28
        assert result.embedding_model == "BAAI/bge-small-en-v1.5"
        assert result.section == "Item 1A - Risk Factors"

    def test_response_dataclass_defaults(self):
        """Test BGEQueryResponse defaults."""
        response = BGEQueryResponse(
            symbol="TEST",
            query="test",
            top_k=5,
            available=False,
        )
        assert response.results == []
        assert response.total_found == 0
        assert response.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])