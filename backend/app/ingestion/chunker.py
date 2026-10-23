from typing import Protocol

from app.config import settings
from app.ingestion.parser import clean_text


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str: ...


def chunk_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into tokenizer-sized overlapping chunks."""
    size = chunk_size if chunk_size is not None else settings.chunk_size_tokens
    overlap_size = overlap if overlap is not None else settings.chunk_overlap_tokens
    if size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap_size < 0 or overlap_size >= size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = clean_text(text)
    if not normalized:
        return []

    token_ids = tokenizer.encode(normalized, add_special_tokens=False)
    if not token_ids:
        return []

    stride = size - overlap_size
    chunks = []
    for start in range(0, len(token_ids), stride):
        chunk = tokenizer.decode(
            token_ids[start : start + size],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(token_ids):
            break
    return chunks
