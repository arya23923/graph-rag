"""
LLM Client — LangChain-powered with Groq (free LLaMA 3) + OpenAI fallback.

Replaces raw HTTP requests with LangChain's ChatGroq / ChatOpenAI wrappers:
  - Automatic retry / rate-limit handling
  - Streaming support
  - Prompt templating via LangChain PromptTemplate
  - Easy model swaps without touching calling code
"""
import os
import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-3.5-turbo"

_INVALID_KEYS = {"your_groq_key_here", "gsk_your_groq_key_here", "your_key_here", ""}


@lru_cache(maxsize=1)
def _get_groq_llm():
    if GROQ_API_KEY in _INVALID_KEYS:
        return None
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY,
                        temperature=0.3, max_tokens=600, request_timeout=20)
    except Exception as e:
        logger.warning("Groq init failed: %s", e)
        return None


@lru_cache(maxsize=1)
def _get_openai_llm():
    if OPENAI_API_KEY in _INVALID_KEYS:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY,
                          temperature=0.3, max_tokens=600, request_timeout=20)
    except Exception as e:
        logger.warning("OpenAI init failed: %s", e)
        return None


_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human",  "{user_prompt}"),
])


def get_llm_answer(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> dict:
    """Try Groq then OpenAI. Returns {answer, provider, success}."""
    for llm, label in [(_get_groq_llm(), "Groq (LLaMA 3)"),
                        (_get_openai_llm(), "OpenAI GPT-3.5")]:
        if llm:
            result = _invoke(llm, system_prompt, user_prompt, label)
            if result["success"]:
                return result
    return {"answer": None, "provider": "none", "success": False}


def _invoke(llm, system_prompt: str, user_prompt: str, provider: str) -> dict:
    try:
        chain = _PROMPT | llm
        resp  = chain.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return {"answer": resp.content.strip(), "provider": provider, "success": True}
    except Exception as e:
        logger.warning("%s invoke failed: %s", provider, e)
        return {"answer": None, "provider": provider, "success": False}


def is_llm_available() -> dict:
    return {"groq": _get_groq_llm() is not None,
            "openai": _get_openai_llm() is not None}


def stream_llm_answer(system_prompt: str, user_prompt: str):
    """Yield tokens from best available LLM (for Streamlit streaming)."""
    llm = _get_groq_llm() or _get_openai_llm()
    if not llm:
        yield "⚠️ No LLM configured. Add GROQ_API_KEY to .env (free at console.groq.com)"
        return
    try:
        chain = _PROMPT | llm
        for chunk in chain.stream({"system_prompt": system_prompt, "user_prompt": user_prompt}):
            if hasattr(chunk, "content"):
                yield chunk.content
    except Exception as e:
        yield f"\n\n⚠️ Streaming error: {e}"
