"""Tests for chunking, retrieval and answer handling (no LLM needed: the LLM call is stubbed).

    python -m pytest -q
"""
import json

import pytest

from rag import answer as answer_mod
from rag.config import EVAL_DIR, TOP_K
from rag.evaluate import keyword_recall
from rag.ingest import clean, load_chunks
from rag.retrieval import Retriever, tokenize


def test_clean_fixes_ligatures_and_hyphenation():
    assert clean("Conﬁguration") == "Configuration"
    assert clean("gener-\native AI") == "generative AI"


def test_chunks_keep_source_page():
    chunks = load_chunks()
    assert len(chunks) > 200
    assert all(c.page >= 1 and c.doc.endswith(".pdf") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_tokenize_drops_stopwords():
    assert tokenize("What is the MAP function?") == ["map", "function"]


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@pytest.mark.parametrize("method", ["keyword", "dense", "hybrid", "rerank"])
def test_retrievers_find_the_four_functions(retriever, method):
    hits = retriever.search("What are the four functions of the AI RMF Core?", method, TOP_K)
    assert len(hits) == TOP_K
    assert any(h.chunk.page in (8, 25) and h.chunk.doc == "NIST.AI.100-1.pdf" for h in hits)


def test_answer_parses_citations_and_declines(retriever, monkeypatch):
    monkeypatch.setattr(answer_mod, "chat", lambda system, user, provider=None: "GOVERN, MAP [1] and MEASURE [2].")
    a = answer_mod.ask("What are the functions?", retriever)
    assert a.cited == [1, 2] and not a.declined and len(a.cited_sources()) == 2
    monkeypatch.setattr(answer_mod, "chat", lambda system, user, provider=None: answer_mod.NOT_FOUND)
    assert answer_mod.ask("Who won the World Cup?", retriever).declined


def test_keyword_recall_supports_alternatives():
    assert keyword_recall("It is a hallucination", ["hallucination|confidently"]) == 1.0
    assert keyword_recall("nothing relevant", ["govern", "map"]) == 0.0


def test_eval_set_is_well_formed():
    d = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    assert len(d["answerable"]) >= 20 and len(d["unanswerable"]) >= 5
    assert all(q["gold"] and q["keywords"] for q in d["answerable"])
