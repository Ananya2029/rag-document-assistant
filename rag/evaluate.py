"""
Evaluate the RAG system on the hand-written question set (eval/questions.json).

    python -m rag.evaluate --retrieval-only     # no LLM needed
    python -m rag.evaluate                      # full: retrieval + answers

Retrieval: hit@k (a gold page is in the top k) and MRR, for BM25 / dense / hybrid.
Answers:   keyword recall (facts mentioned), citation accuracy (cited passage is on a
           gold page), and whether unanswerable questions are correctly declined.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rag.config import EVAL_DIR, REPORT_DIR, TOP_K  # noqa: E402
from rag.retrieval import Retriever  # noqa: E402

METHODS = ["keyword", "dense", "hybrid", "rerank"]


def load_questions():
    return json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))


def is_gold(hit, q):
    return hit.chunk.doc == q["doc"] and hit.chunk.page in q["gold"]


def retrieval_eval(retriever, questions, k=TOP_K):
    results = {}
    for m in METHODS:
        hits_at = {1: [], 3: [], k: []}
        rr = []
        for q in questions:
            hits = retriever.search(q["q"], m, k)
            ranks = [i for i, h in enumerate(hits, 1) if is_gold(h, q)]
            for n in hits_at:
                hits_at[n].append(bool(ranks) and ranks[0] <= n)
            rr.append(1 / ranks[0] if ranks else 0.0)
        results[m] = {f"hit@{n}": round(float(np.mean(v)), 3) for n, v in hits_at.items()} | \
                     {"mrr": round(float(np.mean(rr)), 3)}
    return results


def keyword_recall(text, keywords):
    t = text.lower()
    return float(np.mean([any(alt in t for alt in kw.split("|")) for kw in keywords]))


def answer_eval(retriever, data, method="rerank"):
    from rag.answer import ask
    rows = []
    for q in data["answerable"]:
        a = ask(q["q"], retriever, method)
        cited = a.cited_sources()
        rows.append({"question": q["q"], "type": "answerable", "answer": a.text,
                     "declined": a.declined, "keyword_recall": keyword_recall(a.text, q["keywords"]),
                     "cites_something": bool(cited),
                     "citation_on_gold_page": any(is_gold(h, q) for h in cited),
                     "seconds": round(a.seconds, 1)})
        print(f"  [{len(rows):2d}] recall {rows[-1]['keyword_recall']:.2f} gold-cite {rows[-1]['citation_on_gold_page']!s:5} {a.seconds:5.1f}s  {q['q'][:60]}")
    for q in data["unanswerable"]:
        a = ask(q, retriever, method)
        rows.append({"question": q, "type": "unanswerable", "answer": a.text, "declined": a.declined,
                     "seconds": round(a.seconds, 1)})
        print(f"  [{len(rows):2d}] declined {a.declined!s:5} {a.seconds:5.1f}s  {q[:60]}")
    ans = [r for r in rows if r["type"] == "answerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    summary = {
        "answerable_questions": len(ans),
        "keyword_recall": round(float(np.mean([r["keyword_recall"] for r in ans])), 3),
        "fully_correct_answers": round(float(np.mean([r["keyword_recall"] == 1 for r in ans])), 3),
        "answers_with_citation": round(float(np.mean([r["cites_something"] for r in ans])), 3),
        "citation_accuracy": round(float(np.mean([r["citation_on_gold_page"] for r in ans])), 3),
        "wrongly_declined": round(float(np.mean([r["declined"] for r in ans])), 3),
        "unanswerable_questions": len(una),
        "correctly_declined": round(float(np.mean([r["declined"] for r in una])), 3),
        "median_seconds_per_answer": float(np.median([r["seconds"] for r in rows])),
    }
    return summary, rows


def plot_retrieval(res):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(METHODS))
    for i, metric in enumerate(["hit@1", "hit@3", f"hit@{TOP_K}", "mrr"]):
        ax.bar(x + (i - 1.5) * 0.2, [res[m][metric] for m in METHODS], 0.2, label=metric)
    ax.set_xticks(x, ["BM25\n(keyword)", "Dense\n(embeddings)", "Hybrid\n(RRF)", "Hybrid +\nreranker"])
    ax.set(ylim=(0, 1.05), title="Retrieval quality on the evaluation set")
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "retrieval_comparison.png", dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    REPORT_DIR.mkdir(exist_ok=True)
    data = load_questions()
    retriever = Retriever()

    retrieval = retrieval_eval(retriever, data["answerable"])
    print(json.dumps(retrieval, indent=2))
    plot_retrieval(retrieval)
    out = {"retrieval": retrieval, "n_questions": len(data["answerable"])}

    if not args.retrieval_only:
        from rag.config import LLM_PROVIDER
        from rag.llm import model_name
        t0 = time.time()
        summary, rows = answer_eval(retriever, data)
        out |= {"llm": f"{LLM_PROVIDER}:{model_name()}", "answers": summary,
                "eval_minutes": round((time.time() - t0) / 60, 1)}
        (REPORT_DIR / "answer_details.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(summary, indent=2))
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
