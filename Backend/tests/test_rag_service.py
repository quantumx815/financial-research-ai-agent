import unittest
from unittest.mock import MagicMock, patch

from services.rag_service import (
    RAGService,
    DocumentMetadata,
    RetrievalResult,
    RetrievalResponse,
)


class TestRAGServiceQuery(unittest.TestCase):
    def setUp(self):
        self.service = RAGService()
        self.service._collection = MagicMock()

    def test_empty_query_returns_error(self):
        response = self.service.query("", "AAPL")
        self.assertIsInstance(response, RetrievalResponse)
        self.assertEqual(response.error, "Query text is empty.")
        self.assertEqual(response.results, [])

    def test_whitespace_query_returns_error(self):
        response = self.service.query("   ", "AAPL")
        self.assertEqual(response.error, "Query text is empty.")
        self.assertEqual(response.results, [])

    def test_empty_company_symbol_returns_error(self):
        response = self.service.query("risk", "")
        self.assertEqual(response.error, "Company symbol is empty.")
        self.assertEqual(response.results, [])

    def test_embedding_failure_returns_error(self):
        with patch("services.rag_service.generate_embedding", side_effect=Exception("API down")):
            response = self.service.query("risk", "AAPL")
        self.assertEqual(response.error, "Embedding generation failed: API down")
        self.assertEqual(response.results, [])

    def test_chromadb_failure_returns_error(self):
        with patch("services.rag_service.generate_embedding", return_value=[0.1, 0.2]):
            self.service._collection.query.side_effect = Exception("DB down")
            response = self.service.query("risk", "AAPL")
        self.assertEqual(response.error, "ChromaDB query failed: DB down")
        self.assertEqual(response.results, [])

    def test_successful_query_returns_structured_results(self):
        mock_results = {
            "ids": [["chunk-1"]],
            "documents": [["Apple faces supply chain risks."]],
            "metadatas": [[{
                "company_symbol": "AAPL",
                "document_type": "10-K",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "https://example.com/aapl-10k",
                "document_id": "doc-1",
                "chunk_index": 0,
            }]],
            "distances": [[0.123]],
        }
        self.service._collection.query.return_value = mock_results

        with patch("services.rag_service.generate_embedding", return_value=[0.1, 0.2]):
            response = self.service.query("What are Apple's major financial risks?", "AAPL", n_results=5)

        self.assertIsInstance(response, RetrievalResponse)
        self.assertEqual(response.query, "What are Apple's major financial risks?")
        self.assertEqual(response.company_symbol, "AAPL")
        self.assertEqual(response.top_k, 5)
        self.assertEqual(response.total_found, 1)
        self.assertEqual(len(response.results), 1)
        self.assertIsInstance(response.results[0], RetrievalResult)
        self.assertEqual(response.results[0].chunk_id, "chunk-1")
        self.assertEqual(response.results[0].chunk_text, "Apple faces supply chain risks.")
        self.assertEqual(response.results[0].company_symbol, "AAPL")
        self.assertEqual(response.results[0].document_type, "10-K")
        self.assertEqual(response.results[0].document_year, 2024)
        self.assertEqual(response.results[0].source, "SEC")
        self.assertEqual(response.results[0].source_url, "https://example.com/aapl-10k")
        self.assertEqual(response.results[0].document_id, "doc-1")
        self.assertEqual(response.results[0].chunk_index, 0)
        self.assertEqual(response.results[0].distance, 0.123)

    def test_company_filter_is_applied(self):
        self.service._collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        with patch("services.rag_service.generate_embedding", return_value=[0.1]):
            self.service.query("risk", "TSLA")

        self.service._collection.query.assert_called_once()
        call_kwargs = self.service._collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["where"], {"company_symbol": "TSLA"})

    def test_top_k_is_passed_to_chromadb(self):
        self.service._collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        with patch("services.rag_service.generate_embedding", return_value=[0.1]):
            self.service.query("risk", "NVDA", n_results=10)

        self.service._collection.query.assert_called_once()
        call_kwargs = self.service._collection.query.call_args.kwargs
        self.assertEqual(call_kwargs["n_results"], 10)

    def test_multiple_results_are_returned(self):
        mock_results = {
            "ids": [["c1", "c2", "c3"]],
            "documents": [["text1", "text2", "text3"]],
            "metadatas": [[
                {"company_symbol": "AAPL", "document_type": "10-K", "document_year": 2024, "source": "SEC", "source_url": "", "document_id": "d1", "chunk_index": 0},
                {"company_symbol": "AAPL", "document_type": "10-K", "document_year": 2024, "source": "SEC", "source_url": "", "document_id": "d1", "chunk_index": 1},
                {"company_symbol": "AAPL", "document_type": "10-K", "document_year": 2024, "source": "SEC", "source_url": "", "document_id": "d1", "chunk_index": 2},
            ]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        self.service._collection.query.return_value = mock_results

        with patch("services.rag_service.generate_embedding", return_value=[0.1]):
            response = self.service.query("risk", "AAPL", n_results=3)

        self.assertEqual(len(response.results), 3)
        self.assertEqual(response.total_found, 3)
        self.assertEqual(response.results[0].chunk_id, "c1")
        self.assertEqual(response.results[1].chunk_id, "c2")
        self.assertEqual(response.results[2].chunk_id, "c3")

    def test_missing_optional_fields_are_none(self):
        mock_results = {
            "ids": [["chunk-1"]],
            "documents": [["text"]],
            "metadatas": [[{
                "company_symbol": "AAPL",
                "document_type": "",
                "document_year": 0,
                "source": "SEC",
                "source_url": "",
                "document_id": "",
                "chunk_index": 0,
            }]],
            "distances": [[0.5]],
        }
        self.service._collection.query.return_value = mock_results

        with patch("services.rag_service.generate_embedding", return_value=[0.1]):
            response = self.service.query("risk", "AAPL")

        self.assertEqual(response.results[0].document_year, None)
        self.assertEqual(response.results[0].source_url, None)
        self.assertEqual(response.results[0].chunk_index, 0)

    def test_no_matching_documents_returns_empty_results(self):
        self.service._collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        with patch("services.rag_service.generate_embedding", return_value=[0.1]):
            response = self.service.query("obscure query xyz", "AAPL")

        self.assertEqual(response.results, [])
        self.assertEqual(response.total_found, 0)
        self.assertIsNone(response.error)


class TestRAGServiceDocumentOperations(unittest.TestCase):
    def test_add_document_chunk_stores_metadata(self):
        service = RAGService()
        service._collection = MagicMock()

        metadata = DocumentMetadata(
            company_symbol="AAPL",
            company_name="Apple Inc.",
            document_type="10-K",
            document_year=2024,
            source="SEC",
            source_url="https://example.com",
            document_id="doc-1",
            chunk_index=0,
        )

        with patch("services.rag_service.generate_embedding", return_value=[0.1]):
            service.add_document_chunk("chunk-1", "text", metadata)

        service._collection.add.assert_called_once()
        call_kwargs = service._collection.add.call_args.kwargs
        self.assertEqual(call_kwargs["ids"], ["chunk-1"])
        self.assertEqual(call_kwargs["documents"], ["text"])
        self.assertEqual(call_kwargs["metadatas"][0]["company_symbol"], "AAPL")

    def test_upsert_document_chunk_stores_metadata(self):
        service = RAGService()
        service._collection = MagicMock()

        metadata = DocumentMetadata(
            company_symbol="NVDA",
            company_name="NVIDIA Corporation",
            document_type="10-Q",
            document_year=2024,
            source="SEC",
            source_url="https://example.com",
            document_id="doc-2",
            chunk_index=5,
        )

        with patch("services.rag_service.generate_embedding", return_value=[0.2]):
            service.upsert_document_chunk("chunk-2", "text", metadata)

        service._collection.upsert.assert_called_once()
        call_kwargs = service._collection.upsert.call_args.kwargs
        self.assertEqual(call_kwargs["ids"], ["chunk-2"])
        self.assertEqual(call_kwargs["metadatas"][0]["company_symbol"], "NVDA")
        self.assertEqual(call_kwargs["metadatas"][0]["chunk_index"], 5)


if __name__ == "__main__":
    unittest.main()
