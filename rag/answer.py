"""Retrieve, then answer strictly from the retrieved passages with [n] citations."""
import re
import time
from dataclasses import dataclass, field

from rag.config import LLM_PROVIDER, TOP_K
from rag.llm import chat
from rag.retrieval import Hit, Retriever

NOT_FOUND = "I don't know based on the provided documents."

SYSTEM = """You are a precise assistant that answers questions from numbered document passages.
Use only facts stated in the passages, give the specific details (names, lists, definitions),
and cite the passage number in square brackets after each fact, e.g. [1] or [2][3]."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)   # passage numbers the answer cites
    seconds: float = 0.0

    @property
    def declined(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.text.lower()

    def cited_sources(self) -> list[Hit]:
        return [self.sources[i - 1] for i in self.cited if 0 < i <= len(self.sources)]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.doc}, page {h.chunk.page})\n{h.chunk.text}"
                       for i, h in enumerate(hits, 1))


def ask(question: str, retriever: Retriever, method: str = "hybrid", k: int = TOP_K,
        provider: str = LLM_PROVIDER) -> Answer:
    t0 = time.time()
    hits = retriever.search(question, method, k)
    # Prompt v2: small models over-refused with v1 (48% of answerable questions declined), so the
    # refusal rule is stated once, only for when NO passage is relevant (see README: prompt iterations)
    user = (f"Passages:\n{format_context(hits)}\n\nQuestion: {question}\n\n"
            "Instructions: Most questions CAN be answered from these passages. If any passage is relevant, "
            "answer with the specific facts it gives and cite it after each fact, in the format "
            "\"Bananas are rich in potassium [3].\" (that sentence only shows the citation format). "
            f"Only if none of the passages is relevant, reply exactly: \"{NOT_FOUND}\"")
    text = chat(SYSTEM, user, provider=provider)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return Answer(question, text, hits, cited, time.time() - t0)
