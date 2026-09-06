import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.gemini_service import generate_financial_analysis


class TestGenerateFinancialAnalysis(unittest.TestCase):
    def setUp(self):
        self.company_data = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "current_price": 150.0,
            "market_cap": 2500000000000,
            "currency": "USD",
            "revenue": 365000000000,
            "net_income": 94000000000,
            "eps": 6.0,
            "pe_ratio": 25.0,
            "profit_margin": 0.25,
            "revenue_growth": 0.08,
            "return_on_equity": 0.15,
            "debt_to_equity": 1.5,
            "fifty_two_week_high": 200.0,
            "fifty_two_week_low": 120.0,
        }
        self.news = [
            {
                "title": "Apple releases new iPhone",
                "summary": "Apple announced a new iPhone model.",
                "relevance_score": 5,
                "relevance": "high",
                "publisher": "TechCrunch",
                "published_at": "2024-01-01T00:00:00Z",
                "url": "https://example.com/iphone",
            }
        ]
        self.fixed_response = json.dumps(
            {
                "company_overview": {"summary": "Apple is a technology company."},
                "financial_health": {
                    "overall_assessment": "Strong",
                    "overall_reasoning": "Strong growth and profitability.",
                    "growth": {
                        "assessment": "Strong",
                        "evidence": "Revenue grew 8%.",
                        "significance": "Indicates expansion.",
                    },
                    "profitability": {
                        "assessment": "Strong",
                        "evidence": "Net margin 25%.",
                        "significance": "Efficient conversion.",
                    },
                    "efficiency": {
                        "assessment": "Strong",
                        "evidence": "ROE 15%.",
                        "significance": "Good capital efficiency.",
                    },
                    "valuation": {
                        "assessment": "Neutral",
                        "evidence": "P/E 25.",
                        "significance": "Fair valuation.",
                    },
                    "leverage": {
                        "assessment": "Moderate",
                        "evidence": "D/E 1.5.",
                        "significance": "Manageable leverage.",
                    },
                },
                "recent_news": [
                    {
                        "title": "Apple releases new iPhone",
                        "summary": "Apple announced a new iPhone model.",
                        "relevance": "high",
                        "publisher": "TechCrunch",
                        "published_at": "2024-01-01T00:00:00Z",
                        "url": "https://example.com/iphone",
                    }
                ],
                "opportunities": [
                    {
                        "point": "AI expansion",
                        "evidence": "New AI features.",
                        "why_it_matters": "Could drive growth.",
                    }
                ],
                "risks": [
                    {
                        "point": "Supply chain",
                        "evidence": "China dependence.",
                        "why_it_matters": "Could disrupt production.",
                    }
                ],
                "overall_perspective": {"summary": "Positive outlook."},
                "disclaimer": "This is not financial advice.",
            }
        )

    @patch("services.gemini_service.client.models.generate_content")
    @patch("services.gemini_service.build_research_package")
    def test_prompt_contains_sec_evidence_when_rag_context_has_results(
        self, mock_build_package, mock_generate_content
    ):
        mock_build_package.return_value = {
            "company": {"symbol": "AAPL", "company_name": "Apple Inc."},
            "financial_metrics": {"revenue": 365000000000},
            "recent_news": [{"title": "test"}],
        }
        mock_generate_content.return_value = MagicMock(text=self.fixed_response)

        rag_context = MagicMock()
        rag_context.results = [
            MagicMock(
                document_type="10-K",
                document_year=2024,
                document_id="doc-1",
                chunk_index=5,
                source="SEC",
                source_url="https://example.com/aapl-10k",
                chunk_text="Apple faces supply chain risks.",
            )
        ]

        result = generate_financial_analysis(
            self.company_data, self.news, rag_context=rag_context
        )

        called_prompt = mock_generate_content.call_args.kwargs["contents"]
        self.assertIn("SEC Filing Evidence:", called_prompt)
        self.assertIn("10-K (2024)", called_prompt)
        self.assertIn("Apple faces supply chain risks.", called_prompt)
        self.assertEqual(
            result["company_overview"]["summary"], "Apple is a technology company."
        )

    @patch("services.gemini_service.client.models.generate_content")
    @patch("services.gemini_service.build_research_package")
    def test_prompt_excludes_sec_evidence_when_rag_context_is_none(
        self, mock_build_package, mock_generate_content
    ):
        mock_build_package.return_value = {
            "company": {"symbol": "AAPL", "company_name": "Apple Inc."},
            "financial_metrics": {"revenue": 365000000000},
            "recent_news": [{"title": "test"}],
        }
        mock_generate_content.return_value = MagicMock(text=self.fixed_response)

        result = generate_financial_analysis(self.company_data, self.news)

        called_prompt = mock_generate_content.call_args.kwargs["contents"]
        self.assertNotIn("SEC Filing Evidence:", called_prompt)
        self.assertEqual(
            result["company_overview"]["summary"], "Apple is a technology company."
        )

    @patch("services.gemini_service.client.models.generate_content")
    @patch("services.gemini_service.build_research_package")
    def test_prompt_excludes_sec_evidence_when_rag_context_has_no_results(
        self, mock_build_package, mock_generate_content
    ):
        mock_build_package.return_value = {
            "company": {"symbol": "AAPL", "company_name": "Apple Inc."},
            "financial_metrics": {"revenue": 365000000000},
            "recent_news": [{"title": "test"}],
        }
        mock_generate_content.return_value = MagicMock(text=self.fixed_response)

        rag_context = MagicMock()
        rag_context.results = []

        result = generate_financial_analysis(
            self.company_data, self.news, rag_context=rag_context
        )

        called_prompt = mock_generate_content.call_args.kwargs["contents"]
        self.assertNotIn("SEC Filing Evidence:", called_prompt)
        self.assertEqual(
            result["company_overview"]["summary"], "Apple is a technology company."
        )

    @patch("services.gemini_service.client.models.generate_content")
    @patch("services.gemini_service.build_research_package")
    def test_backward_compatibility_with_two_args(
        self, mock_build_package, mock_generate_content
    ):
        mock_build_package.return_value = {
            "company": {"symbol": "AAPL", "company_name": "Apple Inc."},
            "financial_metrics": {"revenue": 365000000000},
            "recent_news": [{"title": "test"}],
        }
        mock_generate_content.return_value = MagicMock(text=self.fixed_response)

        result = generate_financial_analysis(self.company_data, self.news)

        called_prompt = mock_generate_content.call_args.kwargs["contents"]
        self.assertNotIn("SEC Filing Evidence:", called_prompt)
        self.assertEqual(
            result["company_overview"]["summary"], "Apple is a technology company."
        )

    @patch("services.gemini_service.client.models.generate_content")
    @patch("services.gemini_service.build_research_package")
    def test_output_json_schema_unchanged(
        self, mock_build_package, mock_generate_content
    ):
        mock_build_package.return_value = {
            "company": {"symbol": "AAPL", "company_name": "Apple Inc."},
            "financial_metrics": {"revenue": 365000000000},
            "recent_news": [{"title": "test"}],
        }
        mock_generate_content.return_value = MagicMock(text=self.fixed_response)

        rag_context = MagicMock()
        rag_context.results = [
            MagicMock(
                document_type="10-K",
                document_year=2024,
                document_id="doc-1",
                chunk_index=5,
                source="SEC",
                source_url="https://example.com/aapl-10k",
                chunk_text="Apple faces supply chain risks.",
            )
        ]

        result = generate_financial_analysis(
            self.company_data, self.news, rag_context=rag_context
        )

        expected_keys = {
            "company_overview",
            "financial_health",
            "recent_news",
            "opportunities",
            "risks",
            "overall_perspective",
            "disclaimer",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertNotIn("sec_evidence", result)
        self.assertNotIn("rag_results", result)
        self.assertNotIn("sources", result)


if __name__ == "__main__":
    unittest.main()
