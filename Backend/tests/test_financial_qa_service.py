import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.financial_qa_service import (
    answer_financial_question,
    QAResponse,
    _format_evidence_for_prompt,
    _build_qa_prompt,
    _parse_qa_response,
    _format_evidence_metadata,
)
from services.bge_rag_service import RetrievalResult


class TestFinancialQAService:
    """Unit tests for financial QA service - all external deps mocked."""

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_successful_evidence_grounded_answer(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test successful flow: BGE returns evidence -> Gemini generates answer."""
        # Mock: BGE returns evidence
        mock_results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="The company faces significant market risk due to interest rate changes.",
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
                chunk_text="Revenue increased 15% year over year driven by cloud services.",
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
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = mock_results
        mock_bge_response.total_found = 2
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_response.query = "What are the risks?"
        mock_bge_service.query.return_value = mock_bge_response

        # Mock: Gemini returns grounded answer
        mock_gemini_response = MagicMock()
        mock_gemini_response.text = '{"answer": "Based on Evidence 1, the company faces market risk from interest rate changes. Evidence 2 shows revenue grew 15%.", "sufficient_evidence": true}'
        mock_gemini_client.models.generate_content.return_value = mock_gemini_response

        # Run
        response = answer_financial_question("TEST", "What are the financial risks and revenue performance?")

        # Assert
        assert isinstance(response, QAResponse)
        assert response.symbol == "TEST"
        assert response.query == "What are the financial risks and revenue performance?"
        assert response.available is True
        assert response.error is None
        assert response.evidence_count == 2
        assert "interest rate" in response.answer.lower()
        assert "revenue" in response.answer.lower()

        # Verify evidence metadata preserved
        assert len(response.evidence) == 2
        assert response.evidence[0]["chunk_id"] == "chunk-1"
        assert response.evidence[0]["section"] == "Item 1A - Risk Factors"
        assert response.evidence[0]["evidence_number"] == 1

        # Verify mocks called
        mock_bge_service.query.assert_called_once_with(symbol="TEST", query="What are the financial risks and revenue performance?", top_k=5)
        mock_gemini_client.models.generate_content.assert_called_once()
        # Verify prompt contains evidence
        call_args = mock_gemini_client.models.generate_content.call_args
        assert "SEC FILING EVIDENCE" in call_args.kwargs["contents"]

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_no_evidence_returns_no_gemini_call(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test no evidence -> no Gemini call, clear message."""
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = []
        mock_bge_response.total_found = 0
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_response.query = "test query"
        mock_bge_service.query.return_value = mock_bge_response

        response = answer_financial_question("TEST", "test query")

        assert response.available is True
        assert response.evidence_count == 0
        assert response.answer == "No relevant SEC filing evidence found for this query."
        assert response.error is None

        # Gemini should NOT be called
        mock_gemini_client.models.generate_content.assert_not_called()

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_company_not_available_no_gemini_call(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test company unavailable -> no Gemini call."""
        mock_bge_response = MagicMock()
        mock_bge_response.available = False
        mock_bge_response.error = "Company UNKNOWN not indexed"
        mock_bge_response.symbol = "UNKNOWN"
        mock_bge_response.query = "test query"
        mock_bge_service.query.return_value = mock_bge_response

        response = answer_financial_question("UNKNOWN", "test query")

        assert response.available is False
        assert "not indexed" in response.error
        mock_gemini_client.models.generate_content.assert_not_called()

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_bge_retrieval_failure_no_gemini_call(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test BGE retrieval error -> no Gemini call."""
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.error = "ChromaDB query failed: timeout"
        mock_bge_response.symbol = "TEST"
        mock_bge_service.query.return_value = mock_bge_response

        response = answer_financial_question("TEST", "test query")

        assert response.available is True
        assert "ChromaDB query failed" in response.error
        assert response.evidence_count == 0
        mock_gemini_client.models.generate_content.assert_not_called()

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_gemini_failure_returns_evidence_anyway(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test Gemini failure -> returns evidence but error captured."""
        mock_results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Test evidence",
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
        ]
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = mock_results
        mock_bge_response.total_found = 1
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_response.query = "test query"
        mock_bge_service.query.return_value = mock_bge_response

        mock_gemini_client.models.generate_content.side_effect = Exception("API quota exceeded")

        response = answer_financial_question("TEST", "test query")

        assert response.available is True
        assert response.evidence_count == 1
        assert "Gemini generation failed" in response.error
        assert len(response.evidence) == 1
        assert response.evidence[0]["chunk_id"] == "chunk-1"

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_empty_query_handled(self, mock_bge_service, mock_gemini_client):
        """Test empty query -> passes to BGE which handles it."""
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = []
        mock_bge_response.total_found = 0
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_response.query = ""
        mock_bge_service.query.return_value = mock_bge_response

        response = answer_financial_question("TEST", "")

        assert response.available is True
        assert response.evidence_count == 0
        assert "No relevant SEC filing evidence" in response.answer

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_empty_symbol_handled(self, mock_bge_service, mock_gemini_client):
        """Test empty symbol -> passes to BGE which handles it."""
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = []
        mock_bge_response.total_found = 0
        mock_bge_response.error = None
        mock_bge_response.symbol = ""
        mock_bge_response.query = "test query"
        mock_bge_service.query.return_value = mock_bge_response

        response = answer_financial_question("", "test query")

        assert response.available is True
        assert response.evidence_count == 0
        assert "No relevant SEC filing evidence" in response.answer

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_insufficient_evidence_handled(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test Gemini says evidence insufficient -> clear message."""
        mock_results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Some unrelated text about office supplies.",
                company_symbol="TEST",
                document_type="10-Q",
                document_year=2026,
                source="SEC",
                source_url="https://example.com",
                document_id="doc-001",
                chunk_index=0,
                distance=0.80,
                embedding_model="BAAI/bge-small-en-v1.5",
                section="Item 1A - Risk Factors",
            ),
        ]
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = mock_results
        mock_bge_response.total_found = 1
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_service.query.return_value = mock_bge_response

        mock_gemini_response = MagicMock()
        mock_gemini_response.text = '{"answer": "Not enough info", "sufficient_evidence": false}'
        mock_gemini_client.models.generate_content.return_value = mock_gemini_response

        response = answer_financial_question("TEST", "What are the specific financial risks?")

        assert response.available is True
        assert "insufficient" in response.answer.lower()

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_evidence_metadata_preserved(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test all evidence metadata fields are in response."""
        mock_results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Test evidence text",
                company_symbol="TEST",
                document_type="10-K",
                document_year=2025,
                source="SEC",
                source_url="https://sec.gov/doc-001",
                document_id="0001234567-25-000001",
                chunk_index=3,
                distance=0.28,
                embedding_model="BAAI/bge-small-en-v1.5",
                section="Item 7 - MD&A",
            ),
        ]
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = mock_results
        mock_bge_response.total_found = 1
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_service.query.return_value = mock_bge_response

        mock_gemini_response = MagicMock()
        mock_gemini_response.text = '{"answer": "Test answer", "sufficient_evidence": true}'
        mock_gemini_client.models.generate_content.return_value = mock_gemini_response

        response = answer_financial_question("TEST", "test query")

        assert len(response.evidence) == 1
        ev = response.evidence[0]
        assert ev["evidence_number"] == 1
        assert ev["chunk_id"] == "chunk-1"
        assert ev["document_type"] == "10-K"
        assert ev["document_year"] == 2025
        assert ev["document_id"] == "0001234567-25-000001"
        assert ev["section"] == "Item 7 - MD&A"
        assert ev["chunk_index"] == 3
        assert ev["source"] == "SEC"
        assert ev["source_url"] == "https://sec.gov/doc-001"
        assert ev["distance"] == 0.28
        assert ev["embedding_model"] == "BAAI/bge-small-en-v1.5"

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_prompt_contains_retrieved_evidence(
        self, mock_bge_service, mock_gemini_client
    ):
        """Test that the Gemini prompt includes the retrieved SEC evidence."""
        mock_results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Revenue grew 20% in Q3.",
                company_symbol="TEST",
                document_type="10-Q",
                document_year=2026,
                source="SEC",
                source_url="https://example.com",
                document_id="doc-001",
                chunk_index=0,
                distance=0.30,
                embedding_model="BAAI/bge-small-en-v1.5",
                section="Item 7 - MD&A",
            ),
        ]
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = mock_results
        mock_bge_response.total_found = 1
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_service.query.return_value = mock_bge_response

        mock_gemini_response = MagicMock()
        mock_gemini_response.text = '{"answer": "Revenue grew 20%.", "sufficient_evidence": true}'
        mock_gemini_client.models.generate_content.return_value = mock_gemini_response

        answer_financial_question("TEST", "How did revenue change?")

        call_contents = mock_gemini_client.models.generate_content.call_args.kwargs["contents"]
        assert "Revenue grew 20% in Q3." in call_contents
        assert "Evidence 1" in call_contents
        assert "10-Q" in call_contents
        assert "Item 7 - MD&A" in call_contents
        assert "Answer the user's question based ONLY on the SEC filing evidence" in call_contents

    @patch("services.financial_qa_service._GEMINI_CLIENT")
    @patch("services.financial_qa_service.bge_query_service")
    def test_top_k_passed_to_bge(self, mock_bge_service, mock_gemini_client):
        """Test top_k parameter is passed through to BGE query."""
        mock_bge_response = MagicMock()
        mock_bge_response.available = True
        mock_bge_response.results = []
        mock_bge_response.total_found = 0
        mock_bge_response.error = None
        mock_bge_response.symbol = "TEST"
        mock_bge_service.query.return_value = mock_bge_response

        answer_financial_question("TEST", "test query", top_k=3)

        mock_bge_service.query.assert_called_once_with(symbol="TEST", query="test query", top_k=3)

    def test_format_evidence_for_prompt(self):
        """Test evidence formatting helper."""
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Test text 1",
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
        ]
        formatted = _format_evidence_for_prompt(results)
        
        assert "SEC FILING EVIDENCE" in formatted
        assert "Test text 1" in formatted
        assert "Evidence 1" in formatted
        assert "10-Q" in formatted
        assert "Item 1A - Risk Factors" in formatted

    def test_format_evidence_metadata(self):
        """Test evidence metadata formatting helper."""
        results = [
            RetrievalResult(
                chunk_id="chunk-1",
                chunk_text="Test text",
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
        ]
        metadata = _format_evidence_metadata(results)
        
        assert len(metadata) == 1
        assert metadata[0]["evidence_number"] == 1
        assert metadata[0]["chunk_id"] == "chunk-1"
        assert metadata[0]["section"] == "Item 1A - Risk Factors"

    def test_parse_qa_response(self):
        """Test JSON parsing helper."""
        raw = '{"answer": "Test", "sufficient_evidence": true}'
        parsed = _parse_qa_response(raw)
        assert parsed["answer"] == "Test"
        assert parsed["sufficient_evidence"] is True
        
        # Test with markdown fences
        raw_md = '```json\n{"answer": "Test", "sufficient_evidence": false}\n```'
        parsed = _parse_qa_response(raw_md)
        assert parsed["answer"] == "Test"
        assert parsed["sufficient_evidence"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])