import os
from typing import List, Optional

from sentence_transformers import SentenceTransformer


_BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_BGE_DIMENSION = 384
_MODEL_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "bge",
)

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
        _model = SentenceTransformer(_BGE_MODEL_NAME, cache_folder=_MODEL_CACHE_DIR)
    return _model


def generate_bge_embedding(text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    model = _get_model()
    embedding = model.encode(text.strip(), normalize_embeddings=True)
    return embedding.tolist()


def get_bge_model_info() -> dict:
    return {
        "embedding_provider": "local",
        "embedding_model": _BGE_MODEL_NAME,
        "embedding_dimension": _BGE_DIMENSION,
        "embedding_version": "1.0",
    }


def get_bge_dimension() -> int:
    return _BGE_DIMENSION