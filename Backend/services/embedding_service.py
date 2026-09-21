import os
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()

_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 10.0


def _get_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class QuotaExhaustedError(Exception):
    """Raised when Gemini embedding API reports quota exhaustion."""
    pass


def _is_quota_exhausted(error: Exception) -> bool:
    """Check if error indicates quota/resource exhaustion."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "resource_exhausted",
        "quota",
        "429",
        "rate limit",
    ])


def _is_transient_error(error: Exception) -> bool:
    """Check if error is potentially transient and worth retrying."""
    # Don't retry quota exhaustion
    if _is_quota_exhausted(error):
        return False
    
    error_str = str(error).lower()
    # Retry on server errors, timeouts, etc.
    if any(keyword in error_str for keyword in [
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "unavailable",
        "internal",
    ]):
        return True
    
    # Network errors, timeouts
    return isinstance(error, (TimeoutError, ConnectionError, OSError))


def generate_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    last_error = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            client = _get_client()
            response = client.models.embed_content(
                model=_EMBEDDING_MODEL,
                contents=text.strip(),
            )
            return response.embeddings[0].values
        except Exception as exc:
            last_error = exc
            
            # If quota exhausted, fail fast without retry
            if _is_quota_exhausted(exc):
                raise QuotaExhaustedError(f"Embedding quota exhausted: {exc}") from exc
            
            # If this was the last attempt, don't retry
            if attempt >= _MAX_RETRIES:
                break
            
            # Only retry on transient errors
            if not _is_transient_error(exc):
                raise
            
            # Exponential backoff with jitter
            delay = min(
                _BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5),
                _MAX_DELAY_SECONDS,
            )
            time.sleep(delay)
    
    # All retries exhausted
    raise last_error
