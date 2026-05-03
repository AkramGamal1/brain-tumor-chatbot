"""Load the curated corpus once at startup into a CorpusBundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    body: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CorpusBundle:
    chunks: list[Chunk]
    by_id: dict[str, Chunk]
    crisis_response_text: str


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].lstrip("\n")
    body = text[end + 4 :].lstrip("\n")
    try:
        fm = yaml.safe_load(fm_block) or {}
        if not isinstance(fm, dict):
            return {}, text
    except yaml.YAMLError:
        return {}, text
    return fm, body


def _extract_title_and_body(text: str) -> tuple[str | None, str]:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        body = "\n".join(lines[1:]).lstrip("\n")
        return title, body
    return None, text


def load_corpus(corpus_dir: Path) -> CorpusBundle:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    chunks: list[Chunk] = []
    crisis_text: str | None = None

    for path in sorted(corpus_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, content = _split_frontmatter(raw)
        h1_title, body = _extract_title_and_body(content)
        title = frontmatter.get("title") or h1_title or path.stem.replace("-", " ")
        tags_raw = frontmatter.get("tags", [])
        if isinstance(tags_raw, str):
            tags_raw = [tags_raw]
        tags = tuple(str(t) for t in tags_raw)

        chunk = Chunk(id=path.stem, title=title, body=body.strip(), tags=tags)
        chunks.append(chunk)
        if path.stem == "crisis-resources":
            crisis_text = chunk.body

    if crisis_text is None:
        raise FileNotFoundError(
            "corpus/crisis-resources.md is required for the safety substitution path"
        )

    by_id = {c.id: c for c in chunks}
    return CorpusBundle(chunks=chunks, by_id=by_id, crisis_response_text=crisis_text)
