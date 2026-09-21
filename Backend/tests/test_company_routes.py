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


class TestSecEvidenceResponse(unittest.TestCase):
    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_response_contains_sec_evidence_when_rag_has_results(
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

        result_mock = MagicMock()
        result_mock.document_type = "10-Q"
        result_mock.document_year = 2026
        result_mock.document_id = "0000320193-26-000020"
        result_mock.chunk_index = 106
        result_mock.source = "SEC"
        result_mock.source_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm"
        result_mock.chunk_text = "Apple faces supply chain risks."

        rag_response = MagicMock()
        rag_response.error = None
        rag_response.results = [result_mock]
        mock_rag_service.query.return_value = rag_response

        mock_generate_analysis.return_value = {
            "company_overview": {"summary": "Apple Inc. analysis."}
        }

        response = client.get("/api/company/AAPL/analysis")
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("sec_evidence", data)
        self.assertEqual(len(data["sec_evidence"]), 1)
        self.assertEqual(data["sec_evidence"][0]["document_type"], "10-Q")
        self.assertEqual(data["sec_evidence"][0]["document_year"], 2026)
        self.assertEqual(data["sec_evidence"][0]["document_id"], "0000320193-26-000020")
        self.assertEqual(data["sec_evidence"][0]["chunk_index"], 106)
        self.assertEqual(data["sec_evidence"][0]["source"], "SEC")
        self.assertEqual(data["sec_evidence"][0]["source_url"], "https://www.sec.gov/Archives/edgar/data/320193/000032019326000020/aapl-20260627.htm")
        self.assertEqual(data["sec_evidence"][0]["chunk_text"], "Apple faces supply chain risks.")

    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_response_excludes_sec_evidence_when_rag_has_no_results(
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
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("sec_evidence", data)

    @patch("routes.company.get_company_news")
    @patch("routes.company.get_company_data")
    @patch("routes.company.rag_service")
    @patch("routes.company.generate_financial_analysis")
    def test_response_excludes_sec_evidence_when_rag_has_error(
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
        data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("sec_evidence", data)


if __name__ == "__main__":
    unittest.main()
