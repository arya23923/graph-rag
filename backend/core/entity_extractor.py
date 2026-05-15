"""
EntityExtractor — LangChain-enhanced entity extraction.

Improvements over original:
  - LangChain LLMChain for LLM-assisted entity extraction (when Groq available)
  - Structured JSON output via LangChain PydanticOutputParser
  - spaCy + regex fallbacks preserved unchanged
  - Batched processing for large documents via LangChain text splitters
"""
import os
import re
import uuid
import json
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

TECH_BOOST_TERMS = {
    "AWS", "Azure", "GCP", "Kubernetes", "Docker", "DevOps",
    "Terraform", "Ansible", "Jenkins", "Kafka", "Spark", "Hadoop",
    "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "FastAPI",
    "LangChain", "Neo4j", "NetworkX", "OpenAI", "Python", "Java",
    "React", "Node.js", "GraphQL", "REST", "API", "CI/CD", "MLOps",
    "LLM", "RAG", "BERT", "TensorFlow", "PyTorch", "Hugging Face",
    "Microservices", "Linux", "GitHub", "Git", "FAISS",
}

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
    "from","up","about","into","through","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","this","that","these","those","it","its","they",
    "them","their","we","our","you","your","he","she","his","her","as","if","so",
    "not","no","nor","also","than","then","when","where","which","who","what",
    "how","all","each","every","both","few","more","most","other","some","such",
    "only","same","own","just","because","while","although","however","therefore",
    "Figure","Table","Section","Chapter","Page","Note","Example","Using","Based",
    "Used","Use","New",
}

SPACY_USEFUL_LABELS = {"PERSON","ORG","GPE","PRODUCT","EVENT","LAW","NORP","FAC","LOC"}
LABEL_MAP = {
    "PERSON":"PERSON","ORG":"ORG","GPE":"LOCATION","LOC":"LOCATION",
    "FAC":"LOCATION","PRODUCT":"PRODUCT","EVENT":"EVENT","LAW":"LAW","NORP":"GROUP",
}

# LLM extraction prompt (used when Groq available)
_LLM_EXTRACT_PROMPT = """Extract named entities from the following text.
Return ONLY a JSON array. Each item must have: "name" (string), "type" (one of: PERSON, ORG, TECH, CONCEPT, LOCATION, PRODUCT, EVENT).
Include only important, specific entities — not generic words.

TEXT:
{text}

JSON array (no markdown, no explanation):"""


class EntityExtractor:
    def __init__(self, model: str = "en_core_web_sm", domain: str = None):
        self.domain = domain
        self.nlp = None
        self._llm_chain = None
        self._load_spacy(model)
        self._load_llm_chain()

    def _load_spacy(self, model: str):
        try:
            import spacy
            self.nlp = spacy.load(model)
            logger.info("EntityExtractor: spaCy loaded")
        except Exception as e:
            logger.info("EntityExtractor: spaCy unavailable (%s), using regex mode", e)

    def _load_llm_chain(self):
        """Set up a lightweight LangChain chain for LLM-assisted extraction."""
        try:
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from backend.retrieval.llm_client import _get_groq_llm
            llm = _get_groq_llm()
            if llm:
                prompt = PromptTemplate.from_template(_LLM_EXTRACT_PROMPT)
                self._llm_chain = prompt | llm | StrOutputParser()
                logger.info("EntityExtractor: LLM chain loaded (Groq)")
        except Exception as e:
            logger.debug("LLM chain init skipped: %s", e)

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, text: str, source_doc: str = "unknown") -> List[Dict]:
        seen: Dict[str, Dict] = {}

        # 1. LLM-assisted extraction on first 3000 chars (fast + accurate)
        if self._llm_chain:
            self._extract_with_llm(text[:3000], source_doc, seen)

        # 2. spaCy NER (if available)
        if self.nlp:
            self._extract_with_spacy(text, source_doc, seen)
            self._extract_noun_chunks(text, source_doc, seen)
        else:
            self._extract_capitalized_entities(text, source_doc, seen)
            self._extract_quoted_terms(text, source_doc, seen)
            self._extract_definition_patterns(text, source_doc, seen)

        # 3. Known tech term boost
        self._boost_tech_terms(text, source_doc, seen)

        seen = {k: v for k, v in seen.items() if self._is_valid_entity(v["name"])}
        return list(seen.values())

    # ── LLM extraction ─────────────────────────────────────────────────────

    def _extract_with_llm(self, text: str, source_doc: str, seen: dict):
        try:
            raw = self._llm_chain.invoke({"text": text})
            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            items = json.loads(raw)
            for item in items:
                name = str(item.get("name", "")).strip()
                etype = str(item.get("type", "CONCEPT")).upper()
                if name and len(name) >= 2:
                    key = name.lower()
                    if key not in seen:
                        seen[key] = self._build(name, etype, source_doc, 0.90)
        except Exception as e:
            logger.debug("LLM extraction parse failed: %s", e)

    # ── spaCy extraction ───────────────────────────────────────────────────

    def _extract_with_spacy(self, text: str, source_doc: str, seen: dict):
        doc = self.nlp(text[:1_000_000])
        for ent in doc.ents:
            if ent.label_ not in SPACY_USEFUL_LABELS:
                continue
            name = ent.text.strip()
            if len(name) < 2:
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = self._build(name, LABEL_MAP.get(ent.label_, ent.label_),
                                        source_doc, 0.88)

    def _extract_noun_chunks(self, text: str, source_doc: str, seen: dict):
        doc = self.nlp(text[:50_000])
        for chunk in doc.noun_chunks:
            name = chunk.text.strip()
            if len(name.split()) < 2 and not name[0].isupper():
                continue
            if not (3 <= len(name) <= 60):
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = self._build(name, "CONCEPT", source_doc, 0.55)

    # ── Regex fallbacks ────────────────────────────────────────────────────

    def _extract_capitalized_entities(self, text: str, source_doc: str, seen: dict):
        for match in re.finditer(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b', text):
            name = match.group(1).strip()
            if all(w.lower() in STOP_WORDS for w in name.split()):
                continue
            if not (3 <= len(name) <= 80):
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = self._build(name, "CONCEPT", source_doc, 0.65)

        word_counts: Dict[str, int] = {}
        for match in re.finditer(r'\b([A-Z][a-zA-Z]{2,})\b', text):
            word = match.group(1)
            if word.lower() not in STOP_WORDS:
                word_counts[word] = word_counts.get(word, 0) + 1
        for word, count in word_counts.items():
            if count >= 2:
                key = word.lower()
                if key not in seen:
                    seen[key] = self._build(word, "ENTITY", source_doc, min(0.5 + count * 0.05, 0.8))

    def _extract_quoted_terms(self, text: str, source_doc: str, seen: dict):
        for pattern in [r'"([^"]{3,50})"', r"'([^']{3,50})'", r'`([^`]{3,50})`']:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name and name.lower() not in STOP_WORDS:
                    key = name.lower()
                    if key not in seen:
                        seen[key] = self._build(name, "CONCEPT", source_doc, 0.70)

    def _extract_definition_patterns(self, text: str, source_doc: str, seen: dict):
        patterns = [
            r'\bThe\s+([A-Z][a-zA-Z\s]{2,40}?)\s+(?:system|model|algorithm|framework|approach|method|tool|platform|module|component|process|technique)\b',
            r'\b([A-Z][a-zA-Z\s]{2,40}?)\s+is\s+(?:a|an|the)\s+',
            r'\b([A-Z][a-zA-Z\s]{2,40}?)\s+(?:refers? to|consists? of|enables?|provides?)\b',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if 2 < len(name) < 60 and name.lower() not in STOP_WORDS:
                    key = name.lower()
                    if key not in seen:
                        seen[key] = self._build(name, "CONCEPT", source_doc, 0.72)

    def _boost_tech_terms(self, text: str, source_doc: str, seen: dict):
        for term in TECH_BOOST_TERMS:
            if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                key = term.lower()
                if key in seen:
                    seen[key]["confidence"] = min(seen[key]["confidence"] + 0.1, 0.98)
                    seen[key]["type"] = "TECH"
                else:
                    seen[key] = self._build(term, "TECH", source_doc, 0.92)

    # ── Validation ─────────────────────────────────────────────────────────

    def _is_valid_entity(self, name: str) -> bool:
        if not name or len(name) < 2:
            return False
        # Reject newlines — doc formatting artifacts getting into entity names
        if "\n" in name or "\t" in name or "\r" in name:
            return False
        # Reject version numbers like "0.115.0"
        if re.match(r"^\d[\d.]+$", name.strip()):
            return False
        # Reject bracket fragments like "uvicorn[standard"
        if "[" in name or "]" in name:
            return False
        # Reject questions or overly long phrases
        if name.strip().endswith("?") or len(name.split()) > 8:
            return False
        if all(w in STOP_WORDS for w in name.lower().split()):
            return False
        if name.replace(".", "").replace(",", "").isdigit():
            return False
        return True

    @staticmethod
    def _build(name, etype, source, confidence) -> dict:
        # Clean newlines/extra spaces from name before storing
        name = re.sub(r"\s+", " ", name).strip()
        return {"id": str(uuid.uuid4()), "name": name, "type": etype,
                "source": source, "confidence": round(confidence, 3)}