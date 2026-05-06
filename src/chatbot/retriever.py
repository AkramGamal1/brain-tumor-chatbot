"""Corpus retrieval indirection.

Phase 2.1: prompt builders read the corpus through `Retriever`, never directly
from `CorpusBundle`. `WholeCorpusRetriever` returns the entire corpus
(Phase 1 behavior, byte-identical formatting). `EmbeddingRetriever` is a
Phase 3 stub — the swap point for Option B (semantic retrieval).

The formatted text is computed once at retriever construction so prompt-cache
hashes stay stable across requests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from chatbot.corpus import Chunk, CorpusBundle


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[Chunk, ...]
    formatted_text: str


def _format_chunks(chunks: list[Chunk] | tuple[Chunk, ...]) -> str:
    sections = [f"## {chunk.title}\n\n{chunk.body}" for chunk in chunks]
    return "# Educational corpus\n\n" + "\n\n---\n\n".join(sections)


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str | None = None) -> RetrievalResult:
        """Return chunks (and a formatted block) relevant to `query`.

        `query=None` is the /explain path: no user text, return whatever the
        retriever considers full context. /chat will pass the user message.
        """


class WholeCorpusRetriever(Retriever):
    def __init__(self, corpus: CorpusBundle) -> None:
        self._chunks = tuple(corpus.chunks)
        self._formatted = _format_chunks(self._chunks)
        self._result = RetrievalResult(
            chunks=self._chunks, formatted_text=self._formatted
        )

    def retrieve(self, query: str | None = None) -> RetrievalResult:
        return self._result


class EmbeddingRetriever(Retriever):
    """Phase 3 swap point. Not implemented yet."""

    def retrieve(self, query: str | None = None) -> RetrievalResult:
        raise NotImplementedError(
            "EmbeddingRetriever is a Phase 3 stub. Use WholeCorpusRetriever."
        )
