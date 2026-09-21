import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import generate_embedding, QuotaExhaustedError, _is_quota_exhausted, _is_transient_error


class TestEmbeddingService(unittest.TestCase):
    @patch("services.embedding_service._get_client")
    def test_generate_embedding_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.models.embed_content.return_value = MagicMock(
            embeddings=[MagicMock(values=[0.1, 0.2, 0.3])]
        )
        result = generate_embedding("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_client.models.embed_content.assert_called_once()

    @patch("services.embedding_service._get_client")
    def test_generate_embedding_empty_text_raises(self, mock_get_client):
        with self.assertRaises(ValueError):
            generate_embedding("")
        with self.assertRaises(ValueError):
            generate_embedding("   ")
        mock_get_client.assert_not_called()

    @patch("services.embedding_service._get_client")
    def test_transient_failure_then_success(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.models.embed_content.side_effect = [
            Exception("503 Service Unavailable"),
            MagicMock(embeddings=[MagicMock(values=[0.1, 0.2])]),
        ]
        result = generate_embedding("test text")
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(mock_client.models.embed_content.call_count, 2)

    @patch("services.embedding_service._get_client")
    def test_quota_exhausted_fails_fast_no_retry(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # Simulate quota exhaustion error
        mock_error = Exception("RESOURCE_EXHAUSTED: Quota exhausted")
        mock_client.models.embed_content.side_effect = [mock_error]

        with self.assertRaises(QuotaExhaustedError):
            generate_embedding("test text")
        # Should only call once, no retries
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

    @patch("services.embedding_service._get_client")
    def test_quota_exhausted_in_error_message_fails_fast(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_error = Exception("429 Quota exceeded for embedding model")
        mock_client.models.embed_content.side_effect = [mock_error]

        with self.assertRaises(QuotaExhaustedError):
            generate_embedding("test text")
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

    @patch("services.embedding_service._get_client")
    def test_max_retries_exceeded_raises_last_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_error = Exception("500 Internal Server Error")
        mock_client.models.embed_content.side_effect = [mock_error] * 4  # 3 retries + 1 initial = 4

        with self.assertRaises(Exception) as cm:
            generate_embedding("test text")
        self.assertEqual(str(cm.exception), "500 Internal Server Error")
        # Should call 4 times (1 initial + 3 retries)
        self.assertEqual(mock_client.models.embed_content.call_count, 4)

    @patch("services.embedding_service._get_client")
    def test_non_transient_error_raises_immediately(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_error = Exception("400 Bad Request: Invalid input")
        mock_client.models.embed_content.side_effect = [mock_error]

        with self.assertRaises(Exception) as cm:
            generate_embedding("test text")
        self.assertEqual(str(cm.exception), "400 Bad Request: Invalid input")
        # Should only call once, no retries for non-transient
        self.assertEqual(mock_client.models.embed_content.call_count, 1)

    def test_is_quota_exhausted_detection(self):
        # Test various quota-related error strings
        quota_errors = [
            Exception("RESOURCE_EXHAUSTED"),
            Exception("quota exceeded"),
            Exception("429 Too Many Requests"),
            Exception("rate limit exceeded"),
            Exception("Quota exhausted for project"),
        ]
        for err in quota_errors:
            self.assertTrue(_is_quota_exhausted(err), f"Should detect quota exhaustion: {err}")

        non_quota_errors = [
            Exception("500 Internal Server Error"),
            Exception("503 Service Unavailable"),
            Exception("timeout"),
            Exception("invalid argument"),
        ]
        for err in non_quota_errors:
            self.assertFalse(_is_quota_exhausted(err), f"Should not detect quota exhaustion: {err}")

    def test_is_transient_error_detection(self):
        transient_errors = [
            Exception("500 Internal Server Error"),
            Exception("502 Bad Gateway"),
            Exception("503 Service Unavailable"),
            Exception("504 Gateway Timeout"),
            Exception("timeout"),
            Exception("service unavailable"),
            Exception("internal error"),
            TimeoutError("Connection timeout"),
            ConnectionError("Connection refused"),
            OSError("Network unreachable"),
        ]
        for err in transient_errors:
            self.assertTrue(_is_transient_error(err), f"Should detect transient error: {err}")

        non_transient_errors = [
            Exception("400 Bad Request"),
            Exception("401 Unauthorized"),
            Exception("403 Forbidden"),
            Exception("404 Not Found"),
            Exception("RESOURCE_EXHAUSTED: Quota exhausted"),
            Exception("quota exceeded"),
        ]
        for err in non_transient_errors:
            self.assertFalse(_is_transient_error(err), f"Should not detect transient error: {err}")


if __name__ == "__main__":
    unittest.main()