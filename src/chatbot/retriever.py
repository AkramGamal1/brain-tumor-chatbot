"""Corpus retrieval indirection.

Prompt builders read the corpus through `Retriever`, never directly from
`CorpusBundle`. Two implementations:

- `WholeCorpusRetriever` — returns the entire corpus on every call. Used for
  Phase 1 behavior; prompt-cache friendly because the formatted text is
  byte-stable across requests.

- `EmbeddingRetriever` — local embedding + cosine similarity on page-level
  chunks. Top-k chunks are returned for a given query; if no query is given,
  falls back to the full corpus. Embedding model
  (`sentence-transformers/all-MiniLM-L6-v2`) and embeddings are loaded /
  computed once at construction; no per-request quota cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chatbot.corpus import Chunk, CorpusBundle

if TYPE_CHECKING:  # heavy imports only at type-check time
    import numpy as np
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievalResult:
    chunks: tuple[Chunk, ...]
    formatted_text: str


def _format_chunks(chunks: list[Chunk] | tuple[Chunk, ...]) -> str:
    sections = [f"## {chunk.title}\n\n{chunk.body}" for chunk in chunks]
    return "# Educational corpus\n\n" + "\n\n---\n\n".join(sections)


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str | None = None,
        *,
        always_include_ids: tuple[str, ...] = (),
    ) -> RetrievalResult:
        """Return chunks (and a formatted block) relevant to `query`.

        `query=None` is the /explain path's fallback: no user text — return
        whatever the retriever considers full context. /chat passes the user
        message; /explain passes the predicted class.

        `always_include_ids` is a hint to force-merge specific chunk IDs
        into the result regardless of similarity ranking. Implementations
        that return the whole corpus on every call may ignore it.
        """


class WholeCorpusRetriever(Retriever):
    def __init__(self, corpus: CorpusBundle) -> None:
        self._chunks = tuple(corpus.chunks)
        self._formatted = _format_chunks(self._chunks)
        self._result = RetrievalResult(
            chunks=self._chunks, formatted_text=self._formatted
        )

    def retrieve(
        self,
        query: str | None = None,
        *,
        always_include_ids: tuple[str, ...] = (),
    ) -> RetrievalResult:
        return self._result


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever(Retriever):
    """Page-level semantic retrieval over the corpus.

    Embeddings are computed once at `build()` time (or at first `retrieve`
    call as a safety net). Each retrieve call is one cosine-similarity
    matmul against ~35 vectors — sub-millisecond on CPU.

    `always_include_ids` are chunk IDs that get force-merged with the top-k
    result. Used for `/explain` to guarantee the predicted-class page and
    the model-capability scaffolding are in context regardless of how the
    embedding similarity ranks them.
    """

    def __init__(
        self,
        corpus: CorpusBundle,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 5,
    ) -> None:
        self._chunks: tuple[Chunk, ...] = tuple(corpus.chunks)
        self._chunk_index: dict[str, int] = {c.id: i for i, c in enumerate(self._chunks)}
        self._top_k = top_k
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None
        self._np = None  # set lazily
        self._all_result = RetrievalResult(
            chunks=self._chunks, formatted_text=_format_chunks(self._chunks)
        )

    def build(self) -> None:
        """Eagerly load the model and compute embeddings.

        Call from lifespan startup so the first request does not pay the
        ~2-5 s model-load latency.
        """
        if self._embeddings is not None:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        self._model = SentenceTransformer(self._model_name)
        texts = [f"{c.title}\n\n{c.body}" for c in self._chunks]
        self._embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def retrieve(
        self,
        query: str | None = None,
        *,
        always_include_ids: tuple[str, ...] = (),
    ) -> RetrievalResult:
        self.build()
        assert self._embeddings is not None and self._model is not None and self._np is not None

        if not query or not query.strip():
            return self._all_result

        q_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = (self._embeddings @ q_emb.T).flatten()
        top_idx = self._np.argsort(scores)[::-1][: self._top_k]

        forced_idx = [
            self._chunk_index[i] for i in always_include_ids if i in self._chunk_index
        ]
        merged = list(dict.fromkeys(list(top_idx.tolist()) + forced_idx))
        # Preserve original corpus order for stable formatting
        merged.sort()
        top_chunks = tuple(self._chunks[i] for i in merged)
        return RetrievalResult(
            chunks=top_chunks, formatted_text=_format_chunks(list(top_chunks))
        )
