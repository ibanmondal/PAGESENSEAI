"""Text chunking — turns a cleaned document into retrievable `Chunk` units.

Strategy: recursive paragraph-aware splitter. We split on the largest structural
boundary first (blank lines / headings), then sentences, then hard character cap.
This beats naive fixed-size windowing because it keeps semantically related text
together, which materially improves both BM25 (term co-occurrence) and dense
retrieval (embedding context).
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from ..models.schemas import Chunk

# Heuristics, ordered from coarse to fine. Each regex defines a split boundary.
_PARAGRAPH_SPLIT = re.compile(r"\n\s{0,}\n")  # blank line(s) between blocks
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEADING_HINT = re.compile(r"^(#{1,6}\s|\*\s|-\s|\d+\.\s)", re.MULTILINE)

# Hard floor/ceiling on chunk size so we never produce tiny shards or mega-chunks.
_MIN_CHARS = 80
_MAX_CHARS = 1400


def _split_paragraphs(text: str) -> list[str]:
    """Top-level split on blank lines. Keeps headings attached to their section."""
    parts = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    return parts or [text.strip()]


def _split_sentences(block: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(block) if s.strip()]


def _greedy_pack(
    pieces: Iterable[str], target: int, overlap: int
) -> list[str]:
    """Pack sentence-level pieces into chunks near `target` chars with overlap.

    Overlap is implemented by carrying the last `overlap` chars of the previous
    chunk into the next — gives boundary terms a second chance at retrieval.
    """
    chunks: list[str] = []
    buf = ""
    carry = ""
    for piece in pieces:
        candidate = (carry + piece).strip() if carry else (buf + " " + piece).strip() if buf else piece
        if len(candidate) <= _MAX_CHARS and (len(candidate) <= target or not buf):
            buf = candidate
            continue
        # candidate too big — flush buf, start a new one with overlap
        if buf:
            chunks.append(buf)
            carry = buf[-overlap:] if overlap > 0 else ""
            buf = (carry + " " + piece).strip()
            carry = ""
        else:
            # single piece longer than MAX — hard-split it
            for i in range(0, len(candidate), _MAX_CHARS):
                chunks.append(candidate[i : i + _MAX_CHARS])
            buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= _MIN_CHARS] or chunks


def chunk_document(
    *,
    tab_id: str,
    url: str,
    title: str,
    text: str,
    target_chars: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    """Chunk a single document into `Chunk` objects with stable ids + ordering.

    Args:
        tab_id: owning tab (for provenance/citations).
        url, title: propagated onto every chunk for citation rendering.
        text: cleaned main-content text.
        target_chars: ideal chunk length.
        overlap: char overlap between adjacent chunks.
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    raw_chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= _MAX_CHARS:
            raw_chunks.append(para)
        else:
            # long paragraph — sentence-split then greedily pack
            sentences = _split_sentences(para)
            raw_chunks.extend(_greedy_pack(sentences, target_chars, overlap))

    # Merge tiny trailing fragments into the previous chunk.
    merged: list[str] = []
    for c in raw_chunks:
        if merged and len(c) < _MIN_CHARS:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)

    # Sometimes a single huge paragraph beats MAX_CHARS even after packing.
    final: list[str] = []
    for c in merged:
        if len(c) <= _MAX_CHARS:
            final.append(c)
        else:
            for i in range(0, len(c), _MAX_CHARS):
                final.append(c[i : i + _MAX_CHARS])

    return [
        Chunk(
            chunk_id=f"{tab_id}::{ordinal}",
            tab_id=tab_id,
            url=url,
            title=title,
            text=text_piece,
            ordinal=ordinal,
        )
        for ordinal, text_piece in enumerate(final)
    ]


def merge_adjacent(scored: list, n: int = 1) -> list:
    """Attach neighbouring chunks to a scored hit for richer citation context.

    Used post-retrieval to give the LLM (and the user-facing citation) a bit more
    surrounding text than the single matched chunk. `n` chunks each side.
    """
    return scored  # passthrough; richer context-packing is a later milestone.
