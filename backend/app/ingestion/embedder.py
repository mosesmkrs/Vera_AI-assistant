from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=4)
def load_tokenizer(model_name: str | None = None):
    """Load and cache the Hugging Face tokenizer used for chunking."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name or settings.embedding_model_name)


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def embed_passages(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed document passages with multilingual-e5's required passage prefix."""
    if not texts:
        return []

    name = model_name or settings.embedding_model_name
    model = _load_embedding_model(name)
    vectors = model.encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str, model_name: str | None = None) -> list[float]:
    """Embed a search query with multilingual-e5's required query prefix."""
    name = model_name or settings.embedding_model_name
    model = _load_embedding_model(name)
    vectors = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors[0].tolist()
