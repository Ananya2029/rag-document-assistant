"""
PDF -> page text -> overlapping chunks that remember their source document and page.

    python -m rag.ingest        # (re)builds the index from data/docs/*.pdf
"""
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from rag.config import CHUNK_OVERLAP, CHUNK_WORDS, DOCS_DIR, INDEX_DIR


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    title: str
    page: int          # 1-based PDF page, used in citations
    text: str


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)         # ligatures like "ﬁ" -> "fi" (else keyword search misses them)
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)             # words split across lines
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def doc_title(pdf: pymupdf.Document, path: Path) -> str:
    return (pdf.metadata or {}).get("title") or path.stem


def chunk_pdf(path: Path) -> list[Chunk]:
    pdf = pymupdf.open(path)
    title = doc_title(pdf, path)
    chunks = []
    for page_no, page in enumerate(pdf, start=1):
        words = clean(page.get_text()).split()
        if len(words) < 25:            # skip blank / cover-only pages
            continue
        step = CHUNK_WORDS - CHUNK_OVERLAP
        for i, start in enumerate(range(0, max(len(words) - CHUNK_OVERLAP, 1), step)):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            chunks.append(Chunk(f"{path.stem}-p{page_no}-{i}", path.name, title, page_no, piece))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        chunks += chunk_pdf(pdf)
    return chunks


def main():
    from rag.retrieval import build_index
    chunks = load_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.jsonl").write_text(
        "\n".join(json.dumps(asdict(c)) for c in chunks), encoding="utf-8")
    build_index(chunks)
    docs = sorted({c.doc for c in chunks})
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents: {', '.join(docs)}")


if __name__ == "__main__":
    main()
