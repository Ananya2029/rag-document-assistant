"""Settings. LLM provider and keys come from environment variables / a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", ROOT / "index"))
EVAL_DIR = ROOT / "eval"
REPORT_DIR = ROOT / "reports"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CHUNK_WORDS = 180          # ~250 tokens: small enough to be precise, big enough for context
CHUNK_OVERLAP = 40
TOP_K = 5

# LLM: "ollama" (local, free) or "groq" (free API tier). Switch with LLM_PROVIDER in .env
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   # set in .env, never committed
