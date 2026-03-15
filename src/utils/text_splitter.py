"""
src/utils/text_splitter.py
==========================
Utility for splitting large documents into overlapping chunks.
Used by the condense node for map-reduce processing.
"""

from typing import List


def split_text(text: str, chunk_size: int = 6000, overlap: int = 400) -> List[str]:
    """
    Split `text` into overlapping character-level chunks.

    Args:
        text:       The full document string.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of text chunks. Single-element list if text fits in one chunk.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph boundary for cleaner splits
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind("\n", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]  # drop empty chunks


def estimate_tokens(text: str) -> int:
    """
    Rough token count estimate (1 token ≈ 4 characters for English text).
    Useful for pre-flight checks before sending to the API.
    """
    return len(text) // 4
