import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient

from Backend.main import app

client = TestClient(app)


class TestResearchQueryEndpoint(unittest.TestCase):
    @patch("routes.research.answer_financial_question")
    def test_successful_request(self, mock_qa_service):
        """Test successful QA request with evidence."""
        mock_response = MagicMock()
        mock_response.symbol = "MSFT"
        mock_response.query = "What are Microsoft's major financial risks?"
        mock_response.answer = "Based on SEC filings, Microsoft faces market risk from currency fluctuations and competition in cloud services."
        mock_response.evidence = [
            {
                "evidence_number": 1,
                "chunk_id": "chunk-1",
                "document_type": "10-Q",
                "document_year": 2026,
                "document_id": "0000789019-26-000001",
                "section": "Item 1A - Risk Factors",
                "chunk_index": 5,
                "source": "SEC",
                "source_url": "https://sec.gov/example",
                "distance": 0.25,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
            {
                "evidence_number": 2,
                "chunk_id": "chunk-2",
                "document_type": "10-K",
                "document_year": 2025,
                "document_id": "0000789019-25-000001",
                "section": "Item 7 - MD&A",
                "chunk_index": 12,
                "source": "SEC",
                "source_url": "https://sec.gov/example2",
                "distance": 0.35,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
        ]
        mock_response.evidence_count = 2
        mock_response.available = True
        mock_response.error = None
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "MSFT", "query": "What are Microsoft's major financial risks?", "top_k": 5}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "MSFT")
        self.assertEqual(data["query"], "What are Microsoft's major financial risks?")
        self.assertIn("currency fluctuations", data["answer"])
        self.assertEqual(data["evidence_count"], 2)
        self.assertTrue(data["available"])
        self.assertIsNone(data["error"])
        self.assertEqual(len(data["evidence"]), 2)
        self.assertEqual(data["evidence"][0]["evidence_number"], 1)
        self.assertEqual(data["evidence"][0]["document_type"], "10-Q")
        self.assertEqual(data["evidence"][1]["evidence_number"], 2)
        self.assertEqual(data["evidence"][1]["document_type"], "10-K")

        mock_qa_service.assert_called_once_with(
            symbol="MSFT",
            query="What are Microsoft's major financial risks?",
            top_k=5,
        )

    @patch("routes.research.answer_financial_question")
    def test_default_top_k(self, mock_qa_service):
        """Test default top_k value when not provided."""
        mock_response = MagicMock()
        mock_response.symbol = "AAPL"
        mock_response.query = "test query"
        mock_response.answer = "test answer"
        mock_response.evidence = []
        mock_response.evidence_count = 0
        mock_response.available = True
        mock_response.error = None
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "AAPL", "query": "test query"}
        )

        self.assertEqual(response.status_code, 200)
        mock_qa_service.assert_called_once()
        call_kwargs = mock_qa_service.call_args.kwargs
        self.assertEqual(call_kwargs["top_k"], 5)

    @patch("routes.research.answer_financial_question")
    def test_custom_top_k(self, mock_qa_service):
        """Test custom top_k value is passed through."""
        mock_response = MagicMock()
        mock_response.symbol = "TSLA"
        mock_response.query = "test query"
        mock_response.answer = "test answer"
        mock_response.evidence = []
        mock_response.evidence_count = 0
        mock_response.available = True
        mock_response.error = None
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "TSLA", "query": "test query", "top_k": 10}
        )

        self.assertEqual(response.status_code, 200)
        mock_qa_service.assert_called_once()
        call_kwargs = mock_qa_service.call_args.kwargs
        self.assertEqual(call_kwargs["top_k"], 10)

    def test_invalid_empty_symbol(self):
        """Test validation error for empty symbol."""
        response = client.post(
            "/api/research/query",
            json={"symbol": "", "query": "test query"}
        )

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_invalid_empty_query(self):
        """Test validation error for empty query."""
        response = client.post(
            "/api/research/query",
            json={"symbol": "AAPL", "query": ""}
        )

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_missing_symbol(self):
        """Test validation error for missing symbol field."""
        response = client.post(
            "/api/research/query",
            json={"query": "test query"}
        )

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_missing_query(self):
        """Test validation error for missing query field."""
        response = client.post(
            "/api/research/query",
            json={"symbol": "AAPL"}
        )

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_invalid_top_k_too_low(self):
        """Test validation error for top_k < 1."""
        response = client.post(
            "/api/research/query",
            json={"symbol": "AAPL", "query": "test query", "top_k": 0}
        )

        self.assertEqual(response.status_code, 422)

    def test_invalid_top_k_too_high(self):
        """Test validation error for top_k > 20."""
        response = client.post(
            "/api/research/query",
            json={"symbol": "AAPL", "query": "test query", "top_k": 21}
        )

        self.assertEqual(response.status_code, 422)

    @patch("routes.research.answer_financial_question")
    def test_company_not_available_returns_404(self, mock_qa_service):
        """Test company not in knowledge base returns 404."""
        mock_response = MagicMock()
        mock_response.symbol = "UNKNOWN"
        mock_response.query = "test query"
        mock_response.answer = ""
        mock_response.evidence = []
        mock_response.evidence_count = 0
        mock_response.available = False
        mock_response.error = "Company UNKNOWN not indexed in BGE knowledge base"
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "UNKNOWN", "query": "test query"}
        )

        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("not available in research knowledge base", data["detail"])

    @patch("routes.research.answer_financial_question")
    def test_service_error_returns_400(self, mock_qa_service):
        """Test service-level error returns 400."""
        mock_response = MagicMock()
        mock_response.symbol = "TEST"
        mock_response.query = "test query"
        mock_response.answer = ""
        mock_response.evidence = []
        mock_response.evidence_count = 0
        mock_response.available = True
        mock_response.error = "ChromaDB query failed: timeout"
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "TEST", "query": "test query"}
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("ChromaDB query failed", data["detail"])

    @patch("routes.research.answer_financial_question")
    def test_unexpected_exception_returns_500(self, mock_qa_service):
        """Test unexpected exception returns 500 without stack trace."""
        mock_qa_service.side_effect = Exception("Unexpected internal error")

        response = client.post(
            "/api/research/query",
            json={"symbol": "TEST", "query": "test query"}
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["detail"], "Internal server error during financial research query")

    @patch("routes.research.answer_financial_question")
    def test_no_evidence_found(self, mock_qa_service):
        """Test response when no evidence is found."""
        mock_response = MagicMock()
        mock_response.symbol = "TEST"
        mock_response.query = "test query"
        mock_response.answer = "No relevant SEC filing evidence found for this query."
        mock_response.evidence = []
        mock_response.evidence_count = 0
        mock_response.available = True
        mock_response.error = None
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "TEST", "query": "test query"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["evidence_count"], 0)
        self.assertEqual(len(data["evidence"]), 0)
        self.assertIn("No relevant SEC filing evidence", data["answer"])

    @patch("routes.research.answer_financial_question")
    def test_gemini_failure_returns_evidence_and_error(self, mock_qa_service):
        """Test Gemini failure returns evidence with error message."""
        mock_response = MagicMock()
        mock_response.symbol = "TEST"
        mock_response.query = "test query"
        mock_response.answer = ""
        mock_response.evidence = [
            {
                "evidence_number": 1,
                "chunk_id": "chunk-1",
                "document_type": "10-Q",
                "document_year": 2026,
                "document_id": "doc-001",
                "section": "Item 1A - Risk Factors",
                "chunk_index": 0,
                "source": "SEC",
                "source_url": "https://example.com",
                "distance": 0.30,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
        ]
        mock_response.evidence_count = 1
        mock_response.available = True
        mock_response.error = "Gemini generation failed: API quota exceeded"
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "TEST", "query": "test query"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["evidence_count"], 1)
        self.assertEqual(len(data["evidence"]), 1)
        self.assertIn("Gemini generation failed", data["error"])

    @patch("routes.research.answer_financial_question")
    def test_response_structure_complete(self, mock_qa_service):
        """Test all required fields present in response."""
        mock_response = MagicMock()
        mock_response.symbol = "NVDA"
        mock_response.query = "What is NVDA's revenue?"
        mock_response.answer = "Revenue was $30B."
        mock_response.evidence = [
            {
                "evidence_number": 1,
                "chunk_id": "chunk-1",
                "document_type": "10-Q",
                "document_year": 2026,
                "document_id": "doc-001",
                "section": "Item 7 - MD&A",
                "chunk_index": 0,
                "source": "SEC",
                "source_url": "https://sec.gov",
                "distance": 0.20,
                "embedding_model": "BAAI/bge-small-en-v1.5",
            },
        ]
        mock_response.evidence_count = 1
        mock_response.available = True
        mock_response.error = None
        mock_qa_service.return_value = mock_response

        response = client.post(
            "/api/research/query",
            json={"symbol": "NVDA", "query": "What is NVDA's revenue?"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check all required fields exist
        required_fields = ["symbol", "query", "answer", "evidence", "evidence_count", "available", "error"]
        for field in required_fields:
            self.assertIn(field, data, f"Missing field: {field}")
        
        # Check evidence structure
        self.assertEqual(len(data["evidence"]), 1)
        ev = data["evidence"][0]
        evidence_fields = [
            "evidence_number", "chunk_id", "document_type", "document_year",
            "document_id", "section", "chunk_index", "source", "source_url",
            "distance", "embedding_model"
        ]
        for field in evidence_fields:
            self.assertIn(field, ev, f"Missing evidence field: {field}")


class TestCompanyAnalysisRouteRegression(unittest.TestCase):
    """Ensure existing company analysis endpoint behavior is unchanged."""
    
    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_analysis_with_successful_rag(
        self,
        mock_generate_analysis,
        mock_rag_service,
        mock_get_company_data,
        mock_get_company_news,
    ):
        mock_get_company_data.return_value = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
        }
        mock_get_company_news.return_value = [{"title": "test"}]

        rag_response = MagicMock()
        rag_response.error = None
        rag_response.results = [MagicMock(chunk_text="SEC evidence")]
        mock_rag_service.query.return_value = rag_response

        mock_generate_analysis.return_value = {
            "company_overview": {"summary": "test"}
        }

        response = client.get("/api/company/AAPL/analysis")

        self.assertEqual(response.status_code, 200)
        mock_generate_analysis.assert_called_once()
        call_kwargs = mock_generate_analysis.call_args.kwargs
        self.assertIn("rag_context", call_kwargs)
        self.assertIsNotNone(call_kwargs["rag_context"])

    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_analysis_with_rag_error(
        self,
        mock_generate_analysis,
        mock_rag_service,
        mock_get_company_data,
        mock_get_company_news,
    ):
        mock_get_company_data.return_value = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
        }
        mock_get_company_news.return_value = [{"title": "test"}]

        rag_response = MagicMock()
        rag_response.error = "Embedding generation failed"
        rag_response.results = []
        mock_rag_service.query.return_value = rag_response

        mock_generate_analysis.return_value = {
            "company_overview": {"summary": "test"}
        }

        response = client.get("/api/company/AAPL/analysis")

        self.assertEqual(response.status_code, 200)
        mock_generate_analysis.assert_called_once()
        call_kwargs = mock_generate_analysis.call_args.kwargs
        self.assertIsNone(call_kwargs["rag_context"])

    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_analysis_with_rag_exception(
        self,
        mock_generate_analysis,
        mock_rag_service,
        mock_get_company_data,
        mock_get_company_news,
    ):
        mock_get_company_data.return_value = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
        }
        mock_get_company_news.return_value = [{"title": "test"}]

        mock_rag_service.query.side_effect = Exception("Unexpected error")

        mock_generate_analysis.return_value = {
            "company_overview": {"summary": "test"}
        }

        response = client.get("/api/company/AAPL/analysis")

        self.assertEqual(response.status_code, 200)
        mock_generate_analysis.assert_called_once()
        call_kwargs = mock_generate_analysis.call_args.kwargs
        self.assertIsNone(call_kwargs["rag_context"])

    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_analysis_with_rag_zero_results(
        self,
        mock_generate_analysis,
        mock_rag_service,
        mock_get_company_data,
        mock_get_company_news,
    ):
        mock_get_company_data.return_value = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
        }
        mock_get_company_news.return_value = [{"title": "test"}]

        rag_response = MagicMock()
        rag_response.error = None
        rag_response.results = []
        mock_rag_service.query.return_value = rag_response

        mock_generate_analysis.return_value = {
            "company_overview": {"summary": "test"}
        }

        response = client.get("/api/company/AAPL/analysis")

        self.assertEqual(response.status_code, 200)
        mock_generate_analysis.assert_called_once()
        call_kwargs = mock_generate_analysis.call_args.kwargs
        self.assertIsNone(call_kwargs["rag_context"])


if __name__ == "__main__":
    unittest.main()