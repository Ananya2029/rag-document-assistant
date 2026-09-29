"""Three retrievers over the same chunks: BM25 (keywords), dense (embeddings in ChromaDB), hybrid (RRF)."""
import json
import re
from dataclasses import dataclass
from functools import cached_property

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.config import EMBED_MODEL, INDEX_DIR, RERANK_MODEL, TOP_K
from rag.ingest import Chunk

STOPWORDS = set("a an and are as at be by for from how in is it of on or that the this to what which with "
                "does do can should ai".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


_model = None


def embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def build_index(chunks: list[Chunk]):
    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    if "chunks" in [c.name for c in client.list_collections()]:
        client.delete_collection("chunks")
    col = client.create_collection("chunks", metadata={"hnsw:space": "cosine"})
    vectors = embed([c.text for c in chunks])
    for i in range(0, len(chunks), 500):
        batch = chunks[i:i + 500]
        col.add(ids=[c.chunk_id for c in batch], embeddings=vectors[i:i + 500].tolist(),
                documents=[c.text for c in batch],
                metadatas=[{"doc": c.doc, "title": c.title, "page": c.page} for c in batch])


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, index_dir=INDEX_DIR):
        lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        self.chunks = [Chunk(**json.loads(line)) for line in lines]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.index_dir = index_dir

    @cached_property
    def bm25(self):
        return BM25Okapi([tokenize(c.text) for c in self.chunks])

    @cached_property
    def collection(self):
        return chromadb.PersistentClient(path=str(self.index_dir / "chroma")).get_collection("chunks")

    def keyword(self, query, k=TOP_K) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def dense(self, query, k=TOP_K) -> list[Hit]:
        res = self.collection.query(query_embeddings=embed([query]).tolist(), n_results=k)
        return [Hit(self.by_id[i], 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def hybrid(self, query, k=TOP_K, pool=20, rrf_k=60) -> list[Hit]:
        """Reciprocal rank fusion: rewards chunks that rank well in either retriever."""
        fused = {}
        for hits in (self.keyword(query, pool), self.dense(query, pool)):
            for rank, h in enumerate(hits):
                fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0) + 1 / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.by_id[i], fused[i]) for i in top]

    @cached_property
    def cross_encoder(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANK_MODEL, device="cpu")

    def rerank(self, query, k=TOP_K, pool=20) -> list[Hit]:
        """Hybrid top-`pool`, re-scored by a cross-encoder that reads query and passage together."""
        candidates = self.hybrid(query, pool)
        scores = self.cross_encoder.predict([(query, h.chunk.text) for h in candidates], show_progress_bar=False)
        order = np.argsort(scores)[::-1][:k]
        return [Hit(candidates[i].chunk, float(scores[i])) for i in order]

    def search(self, query, method="hybrid", k=TOP_K) -> list[Hit]:
        return getattr(self, method)(query, k)
