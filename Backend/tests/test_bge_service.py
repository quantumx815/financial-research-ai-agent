import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from services.bge_embedding_service import generate_bge_embedding, get_bge_model_info, get_bge_dimension
from services.bge_rag_service import BGERAGService, DocumentMetadata, RetrievalResult, RetrievalResponse


class TestBGEEmbeddingService:
    def test_embedding_dimension(self):
        assert get_bge_dimension() == 384

    def test_model_info(self):
        info = get_bge_model_info()
        assert info["embedding_provider"] == "local"
        assert info["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert info["embedding_dimension"] == 384
        assert info["embedding_version"] == "1.0"

    def test_generate_embedding_basic(self):
        text = "This is a test sentence for embedding."
        embedding = generate_bge_embedding(text)
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_generate_embedding_normalized(self):
        text = "Test text for normalization check."
        embedding = generate_bge_embedding(text)
        # BGE with normalize_embeddings=True should return unit vectors
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 0.01

    def test_generate_embedding_empty_raises(self):
        with pytest.raises(ValueError):
            generate_bge_embedding("")
        with pytest.raises(ValueError):
            generate_bge_embedding("   ")

    def test_embedding_consistency(self):
        text = "Consistent embedding test."
        emb1 = generate_bge_embedding(text)
        emb2 = generate_bge_embedding(text)
        assert emb1 == emb2


class TestBGERAGService:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.service = BGERAGService()
        # Clean up before each test
        self.service.delete_company_documents("TEST")
        yield
        # Clean up after
        self.service.delete_company_documents("TEST")

    def test_collection_exists(self):
        coll = self.service.get_collection()
        assert coll.name == "financial_documents_bge"
        assert coll.metadata.get("hnsw:space") == "cosine"

    def test_model_info(self):
        info = self.service.get_model_info()
        assert info["embedding_provider"] == "local"
        assert info["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert info["embedding_dimension"] == 384
        assert info["embedding_version"] == "1.0"

    def test_add_and_query_document_chunk(self):
        metadata = DocumentMetadata(
            company_symbol="TEST",
            company_name="Test Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-001",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )

        self.service.add_document_chunk(
            chunk_id="test-chunk-001",
            text="This is a test chunk about financial risks and market conditions.",
            metadata=metadata,
        )

        # Query the chunk
        response = self.service.query(
            query_text="financial risks",
            company_symbol="TEST",
            n_results=5,
        )

        assert response.error is None
        assert response.company_symbol == "TEST"
        assert response.top_k == 5
        assert len(response.results) == 1
        result = response.results[0]
        assert result.chunk_id == "test-chunk-001"
        assert result.company_symbol == "TEST"
        assert result.section == "Item 1A - Risk Factors"
        assert result.distance is not None
        assert result.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_upsert_document_chunk(self):
        metadata = DocumentMetadata(
            company_symbol="TEST",
            company_name="Test Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-002",
            chunk_index=0,
            section="Item 7 - MD&A",
        )

        self.service.upsert_document_chunk(
            chunk_id="test-chunk-002",
            text="Initial text about management discussion.",
            metadata=metadata,
        )

        # Update with new text
        self.service.upsert_document_chunk(
            chunk_id="test-chunk-002",
            text="Updated text about management discussion and analysis.",
            metadata=metadata,
        )

        response = self.service.query(
            query_text="management discussion",
            company_symbol="TEST",
            n_results=5,
        )

        assert len(response.results) == 1
        assert "Updated text" in response.results[0].chunk_text

    def test_company_filtering(self):
        # Add chunks for two companies
        for symbol in ["TESTA", "TESTB"]:
            metadata = DocumentMetadata(
                company_symbol=symbol,
                company_name=f"Test Company {symbol}",
                document_type="10-Q",
                document_year=2026,
                source="SEC",
                source_url="https://example.com",
                document_id=f"test-doc-{symbol}",
                chunk_index=0,
                section="Item 1A - Risk Factors",
            )
            self.service.add_document_chunk(
                chunk_id=f"test-chunk-{symbol}",
                text=f"Risk factors for {symbol}.",
                metadata=metadata,
            )

        # Query TESTA only
        response_a = self.service.query(
            query_text="risk factors",
            company_symbol="TESTA",
            n_results=5,
        )
        assert len(response_a.results) == 1
        assert response_a.results[0].company_symbol == "TESTA"

        # Query TESTB only
        response_b = self.service.query(
            query_text="risk factors",
            company_symbol="TESTB",
            n_results=5,
        )
        assert len(response_b.results) == 1
        assert response_b.results[0].company_symbol == "TESTB"

    def test_empty_query_returns_error(self):
        response = self.service.query(
            query_text="",
            company_symbol="TEST",
            n_results=5,
        )
        assert response.error == "Query text is empty."

    def test_empty_company_returns_error(self):
        response = self.service.query(
            query_text="test query",
            company_symbol="",
            n_results=5,
        )
        assert response.error == "Company symbol is empty."

    def test_metadata_preservation(self):
        metadata = DocumentMetadata(
            company_symbol="TEST",
            company_name="Test Company Inc.",
            document_type="10-K",
            document_year=2025,
            source="SEC",
            source_url="https://sec.gov/test",
            document_id="test-doc-meta",
            chunk_index=5,
            section="Item 1A - Risk Factors",
        )

        self.service.add_document_chunk(
            chunk_id="test-chunk-meta",
            text="Test metadata preservation.",
            metadata=metadata,
        )

        response = self.service.query(
            query_text="test metadata",
            company_symbol="TEST",
            n_results=5,
        )

        result = response.results[0]
        assert result.company_symbol == "TEST"
        assert result.document_type == "10-K"
        assert result.document_year == 2025
        assert result.source == "SEC"
        assert result.source_url == "https://sec.gov/test"
        assert result.document_id == "test-doc-meta"
        assert result.chunk_index == 5
        assert result.section == "Item 1A - Risk Factors"

    def test_distance_scores(self):
        metadata = DocumentMetadata(
            company_symbol="TEST",
            company_name="Test Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-dist",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )

        self.service.add_document_chunk(
            chunk_id="test-chunk-dist-1",
            text="Financial risk factors including market risk and credit risk.",
            metadata=metadata,
        )
        self.service.add_document_chunk(
            chunk_id="test-chunk-dist-2",
            text="Legal proceedings and litigation matters.",
            metadata=metadata,
        )

        response = self.service.query(
            query_text="financial risk market credit",
            company_symbol="TEST",
            n_results=5,
        )

        assert len(response.results) == 2
        # First result should be more relevant (lower distance)
        assert response.results[0].distance <= response.results[1].distance
        assert response.results[0].distance >= 0.0
        assert response.results[0].distance <= 2.0  # Cosine distance range

    def test_top_k_limit(self):
        metadata = DocumentMetadata(
            company_symbol="TEST",
            company_name="Test Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-topk",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )

        for i in range(5):
            self.service.add_document_chunk(
                chunk_id=f"test-chunk-topk-{i}",
                text=f"Test chunk number {i} about financial topics.",
                metadata=metadata,
            )

        response = self.service.query(
            query_text="financial topics",
            company_symbol="TEST",
            n_results=3,
        )

        assert len(response.results) == 3
        assert response.top_k == 3

    def test_list_companies(self):
        metadata = DocumentMetadata(
            company_symbol="TESTLIST",
            company_name="Test List Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-list",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )
        self.service.add_document_chunk(
            chunk_id="test-chunk-list",
            text="Test for list companies.",
            metadata=metadata,
        )

        companies = self.service.list_companies()
        assert "TESTLIST" in companies

    def test_delete_company_documents(self):
        metadata = DocumentMetadata(
            company_symbol="TESTDELETE",
            company_name="Test Delete Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-delete",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )
        self.service.add_document_chunk(
            chunk_id="test-chunk-delete",
            text="Test delete.",
            metadata=metadata,
        )

        # Verify exists
        response = self.service.query(
            query_text="test delete",
            company_symbol="TESTDELETE",
            n_results=5,
        )
        assert len(response.results) == 1

        # Delete
        self.service.delete_company_documents("TESTDELETE")

        # Verify gone
        response = self.service.query(
            query_text="test delete",
            company_symbol="TESTDELETE",
            n_results=5,
        )
        assert len(response.results) == 0

    def test_count(self):
        initial_count = self.service.count()
        metadata = DocumentMetadata(
            company_symbol="TESTCOUNT",
            company_name="Test Count Company",
            document_type="10-Q",
            document_year=2026,
            source="SEC",
            source_url="https://example.com",
            document_id="test-doc-count",
            chunk_index=0,
            section="Item 1A - Risk Factors",
        )
        self.service.add_document_chunk(
            chunk_id="test-chunk-count-1",
            text="Test count 1.",
            metadata=metadata,
        )
        self.service.add_document_chunk(
            chunk_id="test-chunk-count-2",
            text="Test count 2.",
            metadata=metadata,
        )
        assert self.service.count() == initial_count + 2


class TestBGEEmbeddingDimension:
    def test_bge_dimension_is_384(self):
        embedding = generate_bge_embedding("test")
        assert len(embedding) == 384


class TestBGEErrorHandling:
    def test_invalid_input_handling(self):
        with pytest.raises(ValueError):
            generate_bge_embedding(None)

    def test_special_characters(self):
        text = "Test with special chars: ���� & < > \" '"
        embedding = generate_bge_embedding(text)
        assert len(embedding) == 384

    def test_long_text(self):
        text = "This is a test. " * 200  # ~3000 chars
        embedding = generate_bge_embedding(text)
        assert len(embedding) == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])