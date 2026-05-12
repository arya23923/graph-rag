# backend/core/text_extractor.py

import os
from pathlib import Path


class TextExtractor:
    """Extracts plain text from PDF, DOCX, TXT, and Markdown files."""

    ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

    def extract(self, file_path: str) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if ext == ".txt" or ext == ".md":
            return self._extract_from_txt(file_path)
        elif ext == ".pdf":
            return self._extract_from_pdf(file_path)
        elif ext == ".docx":
            return self._extract_from_docx(file_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_from_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _extract_from_pdf(self, file_path: str) -> str:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    def _extract_from_docx(self, file_path: str) -> str:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)