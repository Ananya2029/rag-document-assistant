"""
Chat with your PDFs: answers grounded in the documents, with page citations.

    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.answer import ask  # noqa: E402
from rag.config import DOCS_DIR, INDEX_DIR, LLM_PROVIDER, REPORT_DIR  # noqa: E402
from rag.llm import LLMError, model_name  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def retriever():
    return Retriever()


def rebuild_index():
    from rag.ingest import main as ingest
    ingest()
    st.cache_resource.clear()


with st.sidebar:
    st.header("📚 Documents")
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        st.write(f"• {pdf.name}")
    uploaded = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Index uploaded PDFs"):
        for f in uploaded:
            (DOCS_DIR / Path(f.name).name).write_bytes(f.getvalue())
        with st.spinner("Chunking and embedding..."):
            rebuild_index()
        st.success("Index rebuilt.")
    st.divider()
    method = st.selectbox("Retrieval", ["rerank", "hybrid", "dense", "keyword"],
                          format_func={"rerank": "Hybrid + reranker (best)", "hybrid": "Hybrid (BM25 + dense)",
                                       "dense": "Dense (embeddings)", "keyword": "BM25 (keywords)"}.get)
    st.caption(f"LLM: {LLM_PROVIDER} · {model_name()}")
    ev = REPORT_DIR / "evaluation.json"
    if ev.exists():
        r = json.loads(ev.read_text())["retrieval"]["rerank"]
        st.caption(f"Evaluation: correct page in top 5 for {r['hit@5']:.0%} of test questions")

st.title("📚 Document Q&A Assistant")
st.caption("Answers come only from the indexed documents, with page citations. "
           "If the documents don't contain the answer, the assistant says so.")

if not (INDEX_DIR / "chunks.jsonl").exists():
    st.warning("No index yet — run `python -m rag.ingest` or upload PDFs in the sidebar.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about the documents")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and answering..."):
                a = ask(question, retriever(), method)
            st.markdown(a.text)
            with st.expander(f"Sources ({len(a.sources)} passages · {a.seconds:.1f}s)"):
                for i, h in enumerate(a.sources, 1):
                    mark = "✅ cited" if i in a.cited else ""
                    st.markdown(f"**[{i}] {h.chunk.doc}, page {h.chunk.page}** {mark}")
                    st.caption(h.chunk.text[:600] + ("…" if len(h.chunk.text) > 600 else ""))
            st.session_state.history.append({"role": "assistant", "content": a.text})
        except LLMError as e:
            st.error(str(e))
