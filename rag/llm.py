"""Swappable LLM backends: Ollama (local, free) or Groq (free API tier, OpenAI-compatible)."""
import requests

from rag.config import GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL


class LLMError(RuntimeError):
    pass


def chat(system: str, user: str, provider: str = LLM_PROVIDER, temperature: float = 0.0) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if provider == "ollama":
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=300, json={
                "model": OLLAMA_MODEL, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_ctx": 4096}})
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama not reachable at {OLLAMA_URL} ({e}). Start it with `ollama serve`.") from e
        return r.json()["message"]["content"].strip()
    if provider == "groq":
        if not GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Add it to .env (see .env.example).")
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                          json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature})
        if r.status_code != 200:
            raise LLMError(f"Groq error {r.status_code}: {r.text[:200]}")
        return r.json()["choices"][0]["message"]["content"].strip()
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (use 'ollama' or 'groq').")


def model_name(provider: str = LLM_PROVIDER) -> str:
    return OLLAMA_MODEL if provider == "ollama" else GROQ_MODEL
