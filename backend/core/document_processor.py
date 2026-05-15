"""
DocumentProcessor — LangChain document loading pipeline.

New module (replaces ad-hoc text extraction):
  - LangChain UnstructuredFileLoader / PyPDFLoader / TextLoader auto-dispatch
  - LangChain RecursiveCharacterTextSplitter for smart chunking
  - Returns (full_text, chunks) — full_text for entity extraction,
    chunks available for future vector-store ingestion
"""
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Chunk size tuned for entity/relationship extraction (not too small, not too big)
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150


def load_and_chunk(file_path) -> Dict[str, Any]:
    """
    Load a document and split into overlapping chunks.
    Returns:
        {
            "success": bool,
            "full_text": str,           # full document text
            "chunks": List[str],        # smart chunks for NLP processing
            "metadata": dict,
            "error": str | None,
        }
    """
    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {path}", "full_text": "", "chunks": [], "metadata": {}}

    # Load via LangChain loaders
    docs = _load_documents(path)
    if not docs:
        # Fallback to plain text read
        try:
            full_text = path.read_text(errors="replace")
        except Exception as e:
            return {"success": False, "error": str(e), "full_text": "", "chunks": [], "metadata": {}}
    else:
        full_text = "\n\n".join(d.page_content for d in docs if d.page_content.strip())

    if not full_text.strip():
        return {"success": False, "error": "Document appears to be empty.", "full_text": "", "chunks": [], "metadata": {}}

    # Smart chunking
    chunks = _split_text(full_text)

    return {
        "success":   True,
        "full_text": full_text,
        "chunks":    chunks,
        "metadata":  {
            "filename":   path.name,
            "char_count": len(full_text),
            "chunk_count": len(chunks),
        },
        "error": None,
    }


def _load_documents(path: Path):
    """Dispatch to the right LangChain loader by extension."""
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(str(path)).load()

        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(str(path)).load()

        elif ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()

        else:
            # Generic fallback
            from langchain_community.document_loaders import UnstructuredFileLoader
            return UnstructuredFileLoader(str(path)).load()

    except ImportError as e:
        logger.warning("LangChain loader not available for %s (%s)", ext, e)
        return []
    except Exception as e:
        logger.warning("LangChain loader failed for %s: %s", path.name, e)
        return []


def _split_text(text: str) -> List[str]:
    """Split with LangChain RecursiveCharacterTextSplitter."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        # Naive fallback
        size = CHUNK_SIZE
        return [text[i:i + size] for i in range(0, len(text), size - CHUNK_OVERLAP)]
