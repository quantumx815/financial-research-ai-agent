import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi.testclient import TestClient

from Backend.main import app


client = TestClient(app)


class TestCompanyAnalysisRoute(unittest.TestCase):
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
