import re
import os
from pathlib import Path


class TextExtractor:
    ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

    def extract(self, file_path) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.ALLOWED_EXTENSIONS:
            return {
                "success": False,
                "error": f"Unsupported file type: {ext}",
                "text": "",
                "metadata": {}
            }

        try:
            dispatch = {
                ".txt":  self._from_txt,
                ".md":   self._from_txt,
                ".pdf":  self._from_pdf,
                ".docx": self._from_docx,
            }
            raw = dispatch[ext](path)
            cleaned = self._clean(raw)

            return {
                "success": True,
                "text": cleaned,
                "error": None,
                "metadata": {
                    "filename": path.name,
                    "extension": ext,
                    "char_count": len(cleaned),
                    "word_count": len(cleaned.split()),
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "metadata": {"filename": path.name}
            }

    def _from_txt(self, path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _from_pdf(self, path):
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        except ImportError:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)

    def _from_docx(self, path):
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _clean(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
