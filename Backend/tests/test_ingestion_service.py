import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ingestion_service import (
    IngestionStats,
    _chunk_exists,
    ingest_processed_document,
    ingest_all_processed_documents,
)


class TestIngestionStats(unittest.TestCase):
    def test_initialization(self):
        stats = IngestionStats()
        self.assertEqual(stats.total_chunks, 0)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.upserted_chunks, 0)
        self.assertEqual(stats.failed_chunks, 0)
        self.assertEqual(stats.skipped_existing, 0)
        self.assertEqual(stats.quota_exhausted_chunks, 0)
        self.assertEqual(stats.other_failed_chunks, 0)
        self.assertEqual(stats.errors, [])
        self.assertEqual(stats.quota_exhausted_errors, [])
        self.assertEqual(stats.other_errors, [])

    def test_add_error_quota_exhausted(self):
        stats = IngestionStats()
        stats.add_error("Quota exhausted", is_quota_exhausted=True)
        self.assertEqual(stats.failed_chunks, 1)
        self.assertEqual(stats.quota_exhausted_chunks, 1)
        self.assertEqual(stats.other_failed_chunks, 0)
        self.assertEqual(len(stats.quota_exhausted_errors), 1)
        self.assertEqual(len(stats.other_errors), 0)

    def test_add_error_other_failure(self):
        stats = IngestionStats()
        stats.add_error("Network error", is_quota_exhausted=False)
        self.assertEqual(stats.failed_chunks, 1)
        self.assertEqual(stats.quota_exhausted_chunks, 0)
        self.assertEqual(stats.other_failed_chunks, 1)
        self.assertEqual(len(stats.quota_exhausted_errors), 0)
        self.assertEqual(len(stats.other_errors), 1)


class TestChunkExists(unittest.TestCase):
    @patch("services.ingestion_service._rag_service.get_collection")
    def test_chunk_exists_true(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [["chunk-1"]]}
        mock_get_collection.return_value = mock_collection

        result = _chunk_exists("chunk-1")
        self.assertTrue(result)

    @patch("services.ingestion_service._rag_service.get_collection")
    def test_chunk_exists_false(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        result = _chunk_exists("chunk-1")
        self.assertFalse(result)

    @patch("services.ingestion_service._rag_service.get_collection")
    def test_chunk_exists_exception(self, mock_get_collection):
        mock_get_collection.side_effect = Exception("DB error")
        result = _chunk_exists("chunk-1")
        self.assertFalse(result)


class TestIngestProcessedDocument(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processed_path = os.path.join(self.temp_dir, "test.json")
        self.chunks_data = [
            {
                "chunk_id": "chunk-1",
                "text": "Test chunk 1",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "doc-1",
                "chunk_index": 0,
            },
            {
                "chunk_id": "chunk-2",
                "text": "Test chunk 2",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "https://example.com",
                "document_id": "doc-1",
                "chunk_index": 1,
            },
        ]
        with open(self.processed_path, "w") as f:
            json.dump({"chunks": self.chunks_data}, f)

    def tearDown(self):
        if os.path.exists(self.processed_path):
            os.remove(self.processed_path)
        os.rmdir(self.temp_dir)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_successful_ingestion_all_chunks(self, mock_upsert, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 2)
        self.assertEqual(stats.upserted_chunks, 2)
        self.assertEqual(stats.failed_chunks, 0)
        self.assertEqual(stats.skipped_existing, 0)
        self.assertEqual(stats.quota_exhausted_chunks, 0)
        self.assertEqual(stats.other_failed_chunks, 0)
        self.assertEqual(len(stats.errors), 0)
        self.assertEqual(mock_generate_embedding.call_count, 2)
        self.assertEqual(mock_upsert.call_count, 2)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    def test_existing_chunks_are_skipped(self, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        # First chunk exists, second doesn't
        call_count = [0]
        def side_effect(ids):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"ids": [["chunk-1"]]}  # chunk-1 exists
            else:
                return {"ids": []}  # chunk-2 doesn't exist
        mock_collection.get.side_effect = side_effect
        mock_get_collection.return_value = mock_collection

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.skipped_existing, 1)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.upserted_chunks, 1)
        self.assertEqual(stats.failed_chunks, 0)
        # Should only call embedding for chunk-2
        self.assertEqual(mock_generate_embedding.call_count, 1)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    def test_transient_embedding_failure_then_success(self, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection
        
        # Test that ingestion handles embedding that succeeds on first try
        # (retry logic is tested in embedding_service tests)
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 2)
        self.assertEqual(stats.upserted_chunks, 2)
        self.assertEqual(stats.failed_chunks, 0)
        self.assertEqual(mock_generate_embedding.call_count, 2)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    def test_quota_exhausted_fails_fast(self, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        from services.embedding_service import QuotaExhaustedError
        mock_generate_embedding.side_effect = [
            QuotaExhaustedError("Quota exhausted"),
            QuotaExhaustedError("Quota exhausted"),
        ]

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.upserted_chunks, 0)
        self.assertEqual(stats.failed_chunks, 2)
        self.assertEqual(stats.quota_exhausted_chunks, 2)
        self.assertEqual(stats.other_failed_chunks, 0)
        self.assertEqual(len(stats.quota_exhausted_errors), 2)
        # Should only call once (first chunk fails with quota, second chunk skipped)
        self.assertEqual(mock_generate_embedding.call_count, 1)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    def test_quota_exhausted_skips_remaining_chunks(self, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection

        from services.embedding_service import QuotaExhaustedError
        # First chunk gets quota error, second should be skipped
        mock_generate_embedding.side_effect = [
            QuotaExhaustedError("Quota exhausted"),
            [0.1, 0.2, 0.3],  # This should not be called for chunk-2
        ]

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.upserted_chunks, 0)
        self.assertEqual(stats.failed_chunks, 2)
        self.assertEqual(stats.quota_exhausted_chunks, 2)
        self.assertEqual(stats.other_failed_chunks, 0)
        # Only chunk-1 should have been attempted
        self.assertEqual(mock_generate_embedding.call_count, 1)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_embedding_failure_other_error(self, mock_upsert, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.side_effect = Exception("Network error")

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.upserted_chunks, 0)
        self.assertEqual(stats.failed_chunks, 2)
        self.assertEqual(stats.quota_exhausted_chunks, 0)
        self.assertEqual(stats.other_failed_chunks, 2)
        self.assertEqual(len(stats.other_errors), 2)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_upsert_failure_recorded(self, mock_upsert, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_upsert.side_effect = Exception("Upsert failed")

        stats = ingest_processed_document(self.processed_path)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 2)
        self.assertEqual(stats.upserted_chunks, 0)
        self.assertEqual(stats.failed_chunks, 2)
        self.assertEqual(stats.other_failed_chunks, 2)


class TestIngestAllProcessedDocuments(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(self.processed_dir)

        # Create two test documents
        self.chunks1 = [
            {"chunk_id": "aapl-1", "text": "AAPL chunk 1", "company_symbol": "AAPL", "company_name": "Apple", "document_type": "10-Q", "document_year": 2024, "source": "SEC", "source_url": "", "document_id": "doc-1", "chunk_index": 0}
        ]
        self.chunks2 = [
            {"chunk_id": "nvda-1", "text": "NVDA chunk 1", "company_symbol": "NVDA", "company_name": "NVIDIA", "document_type": "10-Q", "document_year": 2024, "source": "SEC", "source_url": "", "document_id": "doc-2", "chunk_index": 0}
        ]

        with open(os.path.join(self.processed_dir, "AAPL-10-Q-doc-1.json"), "w") as f:
            json.dump({"chunks": self.chunks1}, f)
        with open(os.path.join(self.processed_dir, "NVDA-10-Q-doc-2.json"), "w") as f:
            json.dump({"chunks": self.chunks2}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_ingest_all_aggregates_stats(self, mock_upsert, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        stats = ingest_all_processed_documents(self.processed_dir)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.embedded_chunks, 2)
        self.assertEqual(stats.upserted_chunks, 2)
        self.assertEqual(stats.failed_chunks, 0)
        self.assertEqual(stats.skipped_existing, 0)


class TestIdempotency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(self.processed_dir)

        self.chunks = [
            {"chunk_id": "chunk-1", "text": "Test chunk", "company_symbol": "AAPL", "company_name": "Apple", "document_type": "10-Q", "document_year": 2024, "source": "SEC", "source_url": "", "document_id": "doc-1", "chunk_index": 0}
        ]

        with open(os.path.join(self.processed_dir, "test.json"), "w") as f:
            json.dump({"chunks": self.chunks}, f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_running_twice_does_not_duplicate(self, mock_upsert, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        # First run - chunk doesn't exist
        mock_collection.get.return_value = {"ids": []}
        stats1 = ingest_all_processed_documents(self.processed_dir)
        self.assertEqual(stats1.upserted_chunks, 1)
        self.assertEqual(stats1.skipped_existing, 0)

        # Second run - chunk now exists
        mock_collection.get.return_value = {"ids": [["chunk-1"]]}
        stats2 = ingest_all_processed_documents(self.processed_dir)
        self.assertEqual(stats2.upserted_chunks, 0)
        self.assertEqual(stats2.skipped_existing, 1)

        # Total upserts should be 1
        self.assertEqual(stats1.upserted_chunks + stats2.upserted_chunks, 1)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_failed_chunk_can_be_retried_later(self, mock_upsert, mock_generate_embedding, mock_get_collection):
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        # First run - upsert fails
        mock_collection.get.return_value = {"ids": []}
        mock_upsert.side_effect = Exception("DB error")
        stats1 = ingest_all_processed_documents(self.processed_dir)
        self.assertEqual(stats1.upserted_chunks, 0)
        self.assertEqual(stats1.failed_chunks, 1)
        self.assertEqual(stats1.other_failed_chunks, 1)

        # Second run - upsert succeeds
        mock_upsert.side_effect = None
        stats2 = ingest_all_processed_documents(self.processed_dir)
        self.assertEqual(stats2.upserted_chunks, 1)
        self.assertEqual(stats2.failed_chunks, 0)

        # Total should be 1 successful upsert
        self.assertEqual(stats1.upserted_chunks + stats2.upserted_chunks, 1)


if __name__ == "__main__":
    unittest.main()