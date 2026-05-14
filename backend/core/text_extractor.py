import re
import io
from pathlib import Path

NL = "\n"


class TextExtractor:
    """
    Universal text extractor.
    Supports: .txt, .md, .pdf, .docx, .csv, .json, .xlsx, .xls
    """
    ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md", ".csv", ".json", ".xlsx", ".xls"}

    def extract(self, file_path) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.ALLOWED_EXTENSIONS:
            return {
                "success": False,
                "error": "Unsupported file type: " + ext + ". Supported: " + str(sorted(self.ALLOWED_EXTENSIONS)),
                "text": "",
                "metadata": {}
            }

        try:
            dispatch = {
                ".txt":  self._from_txt,
                ".md":   self._from_txt,
                ".pdf":  self._from_pdf,
                ".docx": self._from_docx,
                ".csv":  self._from_csv,
                ".json": self._from_json,
                ".xlsx": self._from_xlsx,
                ".xls":  self._from_xls,
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

    # ── Readers ─────────────────────────────────────────────────────────────

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
            return NL.join(parts)
        except ImportError:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return NL.join(p.extract_text() or "" for p in reader.pages)

    def _from_docx(self, path):
        from docx import Document
        doc = Document(str(path))
        return NL.join(p.text for p in doc.paragraphs if p.text.strip())

    def _from_csv(self, path):
        import csv
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(str(c) for c in row))
        return NL.join(rows)

    def _from_json(self, path):
        import json
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return self._flatten_json(data)

    def _from_xlsx(self, path):
        import openpyxl
        wb = openpyxl.load_workbook(str(path), read_only=True)
        lines = []
        for ws in wb.worksheets:
            lines.append("Sheet: " + ws.title)
            for row in ws.iter_rows(values_only=True):
                line = ", ".join(str(c) for c in row if c is not None)
                if line.strip():
                    lines.append(line)
        return NL.join(lines)

    def _from_xls(self, path):
        import pandas as pd
        df = pd.read_excel(str(path), engine="xlrd")
        return df.to_string()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _flatten_json(self, obj, depth=0, max_depth=5) -> str:
        if depth > max_depth:
            return str(obj)
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(str(k) + ": " + self._flatten_json(v, depth + 1, max_depth))
            return ". ".join(parts)
        elif isinstance(obj, list):
            return ". ".join(self._flatten_json(item, depth + 1, max_depth) for item in obj[:100])
        else:
            return str(obj)

    def _clean(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()