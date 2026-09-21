import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ingestion_service import (
    IngestionStats,
    MigrationStats,
    _chunk_exists,
    _migration_chunk_exists,
    _get_existing_vector,
    ingest_processed_document,
    ingest_all_processed_documents,
    ingest_migration_chunks,
)
from services.document_processor import compare_processed_documents, ProcessedDocument


class TestIngestionMigration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(self.processed_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_migration_chunk_exists_check(self):
        """Test _migration_chunk_exists returns False for non-existent chunks."""
        result = _migration_chunk_exists("non-existent-id", generation=2)
        self.assertFalse(result)

    def test_migration_stats_initialization(self):
        """Test MigrationStats initializes correctly."""
        stats = MigrationStats()
        self.assertEqual(stats.total_chunks, 0)
        self.assertEqual(stats.migrated_chunks, 0)
        self.assertEqual(stats.skipped_unchanged, 0)
        self.assertEqual(stats.skipped_existing_migration, 0)
        self.assertEqual(stats.failed_chunks, 0)
        self.assertEqual(stats.errors, [])

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_ingest_migration_chunks_only_migrates_changed_and_new(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        mock_collection = MagicMock()
        # Mock get() for _chunk_exists (returns empty for new chunks) and _migration_chunk_exists (returns empty)
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        # Mock get_vector_by_id to return an existing embedding for the changed chunk
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text content",
            "metadata": {"generation": 1, "company_symbol": "AAPL"},
        }

        # Create test data with changed and new chunks
        chunks = [
            {
                "chunk_id": "unchanged-id",
                "text": "Unchanged content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
            {
                "chunk_id": "new-id",
                "text": "Brand new content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 2,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        # Comparison says: unchanged-id is unchanged, changed-new-id is changed, new-id is new
        comparison = {
            "unchanged": ["unchanged-id"],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": ["new-id"],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
            "classifications": {"changed-new-id": "SAFE_REUSE"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 3)
        # Should only migrate changed and new (2 chunks)
        self.assertEqual(stats.migrated_chunks, 2)
        # Should skip unchanged
        self.assertEqual(stats.skipped_unchanged, 1)
        # Should not have errors
        self.assertEqual(stats.failed_chunks, 0)
        # Should have called upsert for 2 chunks
        self.assertEqual(mock_upsert.call_count, 2)
        # Changed chunk should reuse vector (no Gemini call for it)
        # New chunk should generate embedding
        self.assertEqual(mock_generate_embedding.call_count, 1)
        self.assertEqual(stats.reused_vectors, 1)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.safe_reuse, 1)
        self.assertEqual(stats.need_new_embedding, 0)
        self.assertEqual(stats.review, 0)
        self.assertEqual(stats.genuinely_new, 1)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_ingest_migration_chunks_skips_existing_migration(
        self, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that migration skips chunks that already have generation=2."""
        mock_collection = MagicMock()
        # Simulate chunk already existing with generation=2
        mock_collection.get.return_value = {
            "ids": [["changed-new-id"]],
            "metadatas": [[{"generation": 2}]]
        }
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 0)
        self.assertEqual(stats.skipped_existing_migration, 1)
        self.assertEqual(mock_upsert.call_count, 0)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    def test_ingest_migration_chunks_quota_exhaustion(
        self, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test migration handles quota exhaustion."""
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        from services.embedding_service import QuotaExhaustedError
        mock_generate_embedding.side_effect = [
            QuotaExhaustedError("Quota exhausted"),
            QuotaExhaustedError("Quota exhausted"),
        ]

        chunks = [
            {
                "chunk_id": "new-id-1",
                "text": "New content 1.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
            {
                "chunk_id": "new-id-2",
                "text": "New content 2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [],
            "new": ["new-id-1", "new-id-2"],
            "removed": [],
            "mapping": {},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.migrated_chunks, 0)
        self.assertEqual(stats.failed_chunks, 2)
        self.assertEqual(len(stats.errors), 2)
        # Should only attempt first chunk (quota exhausted stops remaining)
        self.assertEqual(mock_generate_embedding.call_count, 1)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_ingest_migration_chunks_passes_generation_and_supersedes(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test migration passes generation=2 and supersedes_chunk_id metadata."""
        mock_collection = MagicMock()
        # Mock get() for _chunk_exists and _migration_chunk_exists (both return empty for new chunk)
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        # Mock get_vector_by_id to return an existing embedding for the changed chunk
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text content",
            "metadata": {"generation": 1, "company_symbol": "AAPL"},
        }

        chunks = [
            {
                "chunk_id": "new-id",
                "text": "New content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-id", "new-id")],
            "new": [],
            "removed": ["old-id"],
            "mapping": {"old-id": "new-id"},
            "classifications": {"new-id": "SAFE_REUSE"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 1)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.safe_reuse, 1)
        # Verify upsert was called with generation=2 and supersedes_chunk_id
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertEqual(call_kwargs.get("supersedes_chunk_id"), "old-id")
        # Should use the reused embedding, not generate new one
        self.assertEqual(call_kwargs.get("embedding"), [0.5, 0.6, 0.7])
        mock_generate_embedding.assert_not_called()


class TestComparisonIntegration(unittest.TestCase):
    """Integration tests for the comparison workflow."""

    def test_full_comparison_workflow(self):
        """Test the complete comparison workflow with realistic data."""
        # Old processed document (simulating current production)
        old_chunks = [
            {
                "chunk_id": "chunk-0",
                "chunk_index": 0,
                "text": "aapl-20260627\nUNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q (Mark One) Quarterly Report...",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "section": None,
            },
            {
                "chunk_id": "chunk-1",
                "chunk_index": 1,
                "text": "Item 1A Risk Factors The company faces various risks including supply chain...",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "section": None,
            },
        ]

        # New processed document (after improved processing)
        new_chunks = [
            {
                "chunk_id": "chunk-0-new",  # Different ID due to filename trimming
                "chunk_index": 0,
                "text": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. 20549 FORM 10-Q (Mark One) Quarterly Report...",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "section": None,
            },
            {
                "chunk_id": "chunk-1",  # Same ID, same content
                "chunk_index": 1,
                "text": "Item 1A Risk Factors The company faces various risks including supply chain...",
                "company_symbol": "AAPL",
                "company_name": "Apple Inc.",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "section": "Item 1A - Risk Factors",  # Now has section!
            },
        ]

        old_doc = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple Inc.", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=2, chunks=old_chunks
        )
        new_doc = ProcessedDocument(
            company_symbol="AAPL", company_name="Apple Inc.", document_type="10-Q",
            document_year=2024, source="SEC", source_url="", document_id="doc-1",
            raw_path="", processed_path="", total_chunks=2, chunks=new_chunks
        )

        comparison = compare_processed_documents(old_doc, new_doc)

        # chunk-1 should be unchanged (same ID, same text)
        self.assertIn("chunk-1", comparison.unchanged)
        # chunk-0 should be detected as changed (similar content, different ID)
        self.assertEqual(len(comparison.changed), 1)
        self.assertIn(("chunk-0", "chunk-0-new"), comparison.changed)
        # No new or removed
        self.assertEqual(len(comparison.new), 0)
        self.assertEqual(len(comparison.removed), 0)


if __name__ == "__main__":
    unittest.main()


class TestVectorReuseMigration(unittest.TestCase):
    """Tests for the vector reuse optimization in migration."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(self.processed_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_changed_chunk_reuses_existing_vector(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that changed chunks reuse existing vectors without calling Gemini."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock get_vector_by_id to return an existing embedding
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text content",
            "metadata": {"generation": 1, "company_symbol": "AAPL"},
        }

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",  # Different text but semantically similar
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
            "classifications": {"changed-new-id": "SAFE_REUSE"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 1)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.safe_reuse, 1)
        self.assertEqual(stats.failed_chunks, 0)
        
        # Gemini should NOT have been called for changed chunk
        mock_generate_embedding.assert_not_called()
        
        # get_vector_by_id should have been called with old chunk ID
        mock_get_vector.assert_called_once_with("old-changed-id")
        
        # upsert should have been called with the reused embedding
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertEqual(call_kwargs.get("supersedes_chunk_id"), "old-changed-id")
        self.assertEqual(call_kwargs.get("embedding"), [0.5, 0.6, 0.7])

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_new_chunk_generates_new_embedding(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that new chunks generate embeddings via Gemini."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = None  # No old vector

        chunks = [
            {
                "chunk_id": "new-id",
                "text": "Brand new content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [],
            "new": ["new-id"],
            "removed": [],
            "mapping": {},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.failed_chunks, 0)
        
        # Gemini SHOULD have been called for new chunk
        mock_generate_embedding.assert_called_once_with("Brand new content.")
        
        # get_vector_by_id should NOT have been called (no supersedes_id)
        mock_get_vector.assert_not_called()
        
        # upsert should have been called with the generated embedding
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertIsNone(call_kwargs.get("supersedes_chunk_id"))
        self.assertEqual(call_kwargs.get("embedding"), [0.1, 0.2, 0.3])

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_changed_chunk_fallback_when_old_vector_missing(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test changed chunk is SKIPPED (not migrated) when old vector not found - no silent Gemini fallback."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.9, 0.8, 0.7]
        mock_get_vector.return_value = None  # Old vector not found

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
            "classifications": {"changed-new-id": "SAFE_REUSE"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 0)  # Not migrated - skipped
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 0)  # No fallback to Gemini
        self.assertEqual(stats.failed_chunks, 1)  # Recorded as failed for retry
        self.assertEqual(stats.safe_reuse, 0)
        
        # Gemini should NOT have been called (no silent fallback)
        mock_generate_embedding.assert_not_called()
        
        # get_vector_by_id should have been called first
        mock_get_vector.assert_called_once_with("old-changed-id")
        
        # upsert should NOT have been called
        mock_upsert.assert_not_called()
        
        # Error should be recorded
        self.assertTrue(any("SAFE_REUSE MISSING" in e for e in stats.errors))

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_removed_chunk_old_vector_untouched(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that removed chunks' old vectors remain untouched in ChromaDB."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text",
            "metadata": {"generation": 1},
        }

        # Only a NEW chunk, not the removed one
        chunks = [
            {
                "chunk_id": "new-id",
                "text": "New content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [],
            "new": ["new-id"],
            "removed": ["removed-old-id"],  # This chunk is NOT in new_chunks
            "mapping": {},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.failed_chunks, 0)
        
        # The removed chunk's vector should NOT have been retrieved
        mock_get_vector.assert_not_called()
        
        # upsert should only be called for the new chunk
        mock_upsert.assert_called_once()

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_generation_2_metadata_correct(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that migrated chunks have correct generation=2 metadata."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text",
            "metadata": {"generation": 1},
        }

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": "Item 1A - Risk Factors",
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.migrated_chunks, 1)
        
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertEqual(call_kwargs.get("supersedes_chunk_id"), "old-changed-id")
        
        # Verify metadata passed includes section
        metadata = call_kwargs.get("metadata")
        self.assertEqual(metadata.section, "Item 1A - Risk Factors")

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_supersedes_chunk_id_correct(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that supersedes_chunk_id correctly references the old chunk."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text",
            "metadata": {"generation": 1},
        }

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.migrated_chunks, 1)
        
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("supersedes_chunk_id"), "old-changed-id")

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_repeated_migration_skips_already_migrated(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that re-running migration skips chunks already migrated to generation=2."""
        mock_collection = MagicMock()
        # First call: check _migration_chunk_exists - returns generation=2 exists
        mock_collection.get.return_value = {
            "ids": [["changed-new-id"]],
            "metadatas": [[{"generation": 2}]]
        }
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text",
            "metadata": {"generation": 1},
        }

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 0)
        self.assertEqual(stats.skipped_existing_migration, 1)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 0)
        
        mock_generate_embedding.assert_not_called()
        mock_get_vector.assert_not_called()
        mock_upsert.assert_not_called()

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_changed_chunk_need_new_embedding_calls_gemini(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that NEED_NEW_EMBEDDING classification generates a new embedding via Gemini."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text content",
            "metadata": {"generation": 1, "company_symbol": "AAPL"},
        }

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
            "classifications": {"changed-new-id": "NEED_NEW_EMBEDDING"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.need_new_embedding, 1)
        self.assertEqual(stats.safe_reuse, 0)
        self.assertEqual(stats.failed_chunks, 0)
        
        # Gemini SHOULD have been called
        mock_generate_embedding.assert_called_once()
        
        # get_vector_by_id should NOT be called for NEED_NEW_EMBEDDING
        mock_get_vector.assert_not_called()
        
        # upsert should have been called with the generated embedding
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertEqual(call_kwargs.get("supersedes_chunk_id"), "old-changed-id")
        self.assertEqual(call_kwargs.get("embedding"), [0.1, 0.2, 0.3])

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_changed_chunk_review_calls_gemini(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that REVIEW classification generates a new embedding via Gemini."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = {
            "embedding": [0.5, 0.6, 0.7],
            "document": "Old text content",
            "metadata": {"generation": 1, "company_symbol": "AAPL"},
        }

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
            "classifications": {"changed-new-id": "REVIEW"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.review, 1)
        self.assertEqual(stats.safe_reuse, 0)
        self.assertEqual(stats.failed_chunks, 0)
        
        # Gemini SHOULD have been called
        mock_generate_embedding.assert_called_once()
        
        # get_vector_by_id should NOT be called for REVIEW
        mock_get_vector.assert_not_called()
        
        # upsert should have been called with the generated embedding
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertEqual(call_kwargs.get("supersedes_chunk_id"), "old-changed-id")
        self.assertEqual(call_kwargs.get("embedding"), [0.1, 0.2, 0.3])

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_new_chunk_calls_gemini(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that GENUINELY_NEW classification generates a new embedding via Gemini."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_get_vector.return_value = None  # No old vector

        chunks = [
            {
                "chunk_id": "new-id",
                "text": "Brand new content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [],
            "new": ["new-id"],
            "removed": [],
            "mapping": {},
            "classifications": {},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 1)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.genuinely_new, 1)
        self.assertEqual(stats.failed_chunks, 0)
        
        # Gemini SHOULD have been called for new chunk
        mock_generate_embedding.assert_called_once_with("Brand new content.")
        
        # get_vector_by_id should NOT have been called (no supersedes_id)
        mock_get_vector.assert_not_called()
        
        # upsert should have been called with the generated embedding
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        self.assertEqual(call_kwargs.get("generation"), 2)
        self.assertIsNone(call_kwargs.get("supersedes_chunk_id"))
        self.assertEqual(call_kwargs.get("embedding"), [0.1, 0.2, 0.3])

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_safe_reuse_missing_old_vector_skipped_safely(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test SAFE_REUSE with missing old vector is skipped safely without Gemini fallback."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.9, 0.8, 0.7]
        mock_get_vector.return_value = None  # Old vector not found

        chunks = [
            {
                "chunk_id": "changed-new-id",
                "text": "Changed content v2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [("old-changed-id", "changed-new-id")],
            "new": [],
            "removed": ["old-changed-id"],
            "mapping": {"old-changed-id": "changed-new-id"},
            "classifications": {"changed-new-id": "SAFE_REUSE"},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.migrated_chunks, 0)
        self.assertEqual(stats.reused_vectors, 0)
        self.assertEqual(stats.embedded_chunks, 0)
        self.assertEqual(stats.safe_reuse, 0)
        self.assertEqual(stats.failed_chunks, 1)
        
        # Gemini should NOT have been called (no silent fallback)
        mock_generate_embedding.assert_not_called()
        
        # get_vector_by_id should have been called first
        mock_get_vector.assert_called_once_with("old-changed-id")
        
        # upsert should NOT have been called
        mock_upsert.assert_not_called()
        
        # Error should be recorded
        self.assertTrue(any("SAFE_REUSE MISSING" in e for e in stats.errors))

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_quota_exhaustion_stops_migration(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that quota exhaustion on new chunk stops remaining migration."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        from services.embedding_service import QuotaExhaustedError
        mock_generate_embedding.side_effect = QuotaExhaustedError("Quota exhausted")
        mock_get_vector.return_value = None

        chunks = [
            {
                "chunk_id": "new-id-1",
                "text": "New content 1.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
            {
                "chunk_id": "new-id-2",
                "text": "New content 2.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 1,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        comparison = {
            "unchanged": [],
            "changed": [],
            "new": ["new-id-1", "new-id-2"],
            "removed": [],
            "mapping": {},
        }

        stats = ingest_migration_chunks(processed_path, comparison)

        self.assertEqual(stats.total_chunks, 2)
        self.assertEqual(stats.migrated_chunks, 0)
        self.assertEqual(stats.failed_chunks, 2)
        self.assertEqual(len(stats.errors), 2)
        
        # Should only attempt first chunk (quota exhausted stops remaining)
        self.assertEqual(mock_generate_embedding.call_count, 1)
        
        # Second chunk should be skipped due to quota exhaustion
        self.assertTrue(any("quota exhaustion" in e.lower() for e in stats.errors))

    @patch("services.ingestion_service._rag_service.get_collection")
    @patch("services.ingestion_service.generate_embedding")
    @patch("services.ingestion_service._rag_service.upsert_document_chunk")
    @patch("services.ingestion_service._rag_service.get_vector_by_id")
    def test_normal_ingestion_unchanged(
        self, mock_get_vector, mock_upsert, mock_generate_embedding, mock_get_collection
    ):
        """Test that normal ingestion (ingest_processed_document) behavior is unchanged."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_get_collection.return_value = mock_collection
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3]

        chunks = [
            {
                "chunk_id": "normal-id",
                "text": "Normal ingestion content.",
                "company_symbol": "AAPL",
                "company_name": "Apple",
                "document_type": "10-Q",
                "document_year": 2024,
                "source": "SEC",
                "source_url": "",
                "document_id": "doc-1",
                "chunk_index": 0,
                "section": None,
            },
        ]

        processed_path = os.path.join(self.processed_dir, "test.json")
        with open(processed_path, "w") as f:
            json.dump({"chunks": chunks}, f)

        # Use normal ingestion, not migration
        from services.ingestion_service import ingest_processed_document
        stats = ingest_processed_document(processed_path)

        self.assertEqual(stats.total_chunks, 1)
        self.assertEqual(stats.embedded_chunks, 1)
        self.assertEqual(stats.upserted_chunks, 1)
        self.assertEqual(stats.skipped_existing, 0)
        self.assertEqual(stats.failed_chunks, 0)
        
        # Normal ingestion calls upsert_document_chunk WITHOUT generation/supersedes
        mock_upsert.assert_called_once()
        call_kwargs = mock_upsert.call_args.kwargs
        # generation not explicitly passed -> uses default (1), so not in kwargs
        self.assertIsNone(call_kwargs.get("generation"))
        self.assertIsNone(call_kwargs.get("supersedes_chunk_id"))
        self.assertIsNone(call_kwargs.get("embedding"))  # Should NOT pass explicit embedding