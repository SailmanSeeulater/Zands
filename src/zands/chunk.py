"""Split article text into ~300-800-word chunks on paragraph boundaries, with a small overlap."""

from __future__ import annotations

import re

MIN_WORDS = 300
MAX_WORDS = 800
OVERLAP_WORDS = 40

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")


def _words(text: str) -> int:
    return len(text.split())


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs from plain text. Blank lines separate paragraphs; if there
    are none (trafilatura emits one paragraph per line), single newlines do."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    sep = r"\n\s*\n" if re.search(r"\n\s*\n", text) else r"\n"
    paras = [" ".join(p.split()) for p in re.split(sep, text)]
    return [p for p in paras if p]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Break a paragraph longer than max_words into sentence groups (and,
    for a single monster sentence, plain word windows)."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(paragraph):
        n = _words(sentence)
        if n > max_words:
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            words = sentence.split()
            pieces.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))
            continue
        if count + n > max_words and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def _overlap_tail(units: list[str], overlap_words: int) -> list[str]:
    """Trailing whole sentences of the previous chunk, up to overlap_words."""
    if overlap_words <= 0 or not units:
        return []
    tail: list[str] = []
    count = 0
    for sentence in reversed(split_sentences(units[-1])):
        n = _words(sentence)
        if count + n > overlap_words:
            break
        tail.insert(0, sentence)
        count += n
    return [" ".join(tail)] if tail else []


def chunk_text(
    text: str,
    min_words: int = MIN_WORDS,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    """Chunk text greedily by paragraph.

    A chunk is closed when the next paragraph would push it past max_words.
    The next chunk starts with the last sentence(s) of the previous one (up to
    overlap_words) so a fact split across the boundary is still findable.
    A short final chunk (< min_words) is merged into the previous one when
    that stays within max_words. Text shorter than min_words is one chunk.
    Paragraphs are separated by blank lines in the output.
    """
    units: list[str] = []
    for para in split_paragraphs(text):
        units.extend(_split_long(para, max_words) if _words(para) > max_words else [para])
    if not units:
        return []

    chunks: list[list[str]] = []
    prefixes: list[int] = []  # how many leading units of each chunk are overlap
    current: list[str] = []
    prefix = 0
    count = 0
    for unit in units:
        n = _words(unit)
        if len(current) > prefix and count + n > max_words:
            chunks.append(current)
            prefixes.append(prefix)
            current = _overlap_tail(current, overlap_words)
            count = sum(_words(u) for u in current)
            if count + n > max_words:  # overlap would overflow; drop it
                current, count = [], 0
            prefix = len(current)
        current.append(unit)
        count += n
    if len(current) > prefix:
        chunks.append(current)
        prefixes.append(prefix)

    if len(chunks) > 1:
        fresh_tail = chunks[-1][prefixes[-1]:]
        fresh_words = sum(_words(u) for u in fresh_tail)
        if fresh_words < min_words and sum(_words(u) for u in chunks[-2]) + fresh_words <= max_words:
            chunks[-2] = chunks[-2] + fresh_tail
            chunks.pop()

    return ["\n\n".join(c) for c in chunks]
