"""
EntityExtractor — extracts real entities from ANY uploaded document.
Works WITHOUT spaCy using smart regex + capitalization heuristics.
With spaCy: even better NER (persons, orgs, locations, products).
"""
import os
import re
import uuid
from typing import List, Dict

# Tech terms get confidence boost if found
TECH_BOOST_TERMS = {
    "AWS", "Azure", "GCP", "Kubernetes", "Docker", "DevOps",
    "Terraform", "Ansible", "Jenkins", "Kafka", "Spark", "Hadoop",
    "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "FastAPI",
    "LangChain", "Neo4j", "NetworkX", "OpenAI", "Python", "Java",
    "React", "Node.js", "GraphQL", "REST", "API", "CI/CD", "MLOps",
    "LLM", "RAG", "BERT", "TensorFlow", "PyTorch", "Hugging Face",
    "Microservices", "Kubernetes", "Linux", "GitHub", "Git",
}

# Words to skip — too generic, not useful as entities
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "through",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "as", "if", "so", "not", "no", "nor",
    "also", "than", "then", "when", "where", "which", "who", "what",
    "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "only", "same", "own", "just", "because",
    "while", "although", "however", "therefore", "thus", "hence",
    "Figure", "Table", "Section", "Chapter", "Page", "Note",
    "Example", "Using", "Based", "Used", "Use", "Used", "New",
}

SPACY_USEFUL_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "NORP", "FAC", "LOC"}
LABEL_MAP = {
    "PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION",
    "LOC": "LOCATION", "FAC": "LOCATION", "PRODUCT": "PRODUCT",
    "EVENT": "EVENT", "LAW": "LAW", "NORP": "GROUP",
}


class EntityExtractor:

    def __init__(self, model: str = "en_core_web_sm", domain: str = None):
        self.domain = domain
        self.nlp = None
        self._load_spacy(model)

    def _load_spacy(self, model: str):
        try:
            import spacy
            self.nlp = spacy.load(model)
            print("[EntityExtractor] spaCy loaded — full NER active")
        except Exception as e:
            print(f"[EntityExtractor] spaCy not available ({e}). Using smart regex mode.")

    # ── Public API ─────────────────────────────────────────────────────────

    def extract(self, text: str, source_doc: str = "unknown") -> List[Dict]:
        """
        Extract real entities from any document text.
        Returns entities actually found IN the text — not hardcoded lists.
        """
        seen: Dict[str, Dict] = {}

        if self.nlp:
            # Best mode: spaCy full NER
            self._extract_with_spacy(text, source_doc, seen)
            self._extract_noun_chunks(text, source_doc, seen)
        else:
            # Smart fallback: regex-based extraction from actual document text
            self._extract_capitalized_entities(text, source_doc, seen)
            self._extract_quoted_terms(text, source_doc, seen)
            self._extract_definition_patterns(text, source_doc, seen)

        # Always: boost known tech terms if they appear in text
        self._boost_tech_terms(text, source_doc, seen)

        # Filter noise
        seen = {k: v for k, v in seen.items() if self._is_valid_entity(v["name"])}

        return list(seen.values())

    # ── spaCy extraction ───────────────────────────────────────────────────

    def _extract_with_spacy(self, text: str, source_doc: str, seen: dict):
        doc = self.nlp(text[:1000000])
        for ent in doc.ents:
            if ent.label_ not in SPACY_USEFUL_LABELS:
                continue
            name = ent.text.strip()
            if len(name) < 2:
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = self._build(
                    name=name,
                    etype=LABEL_MAP.get(ent.label_, ent.label_),
                    source=source_doc,
                    confidence=0.88,
                )

    def _extract_noun_chunks(self, text: str, source_doc: str, seen: dict):
        doc = self.nlp(text[:50000])
        for chunk in doc.noun_chunks:
            name = chunk.text.strip()
            if len(name.split()) < 2 and not name[0].isupper():
                continue
            if not (3 <= len(name) <= 60):
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = self._build(name, "CONCEPT", source_doc, 0.55)

    # ── Smart regex fallback (no spaCy) ────────────────────────────────────

    def _extract_capitalized_entities(self, text: str, source_doc: str, seen: dict):
        """
        Extract Capitalized Words and Multi-Word Proper Nouns from actual document.
        This finds real names/terms from YOUR document, not a hardcoded list.
        """
        # Multi-word proper nouns: "Machine Learning", "Natural Language Processing"
        multi_pattern = r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b'
        for match in re.finditer(multi_pattern, text):
            name = match.group(1).strip()
            words = name.split()
            # Skip if all words are stop words
            if all(w.lower() in STOP_WORDS for w in words):
                continue
            if len(name) < 3 or len(name) > 80:
                continue
            key = name.lower()
            if key not in seen:
                seen[key] = self._build(name, "CONCEPT", source_doc, 0.65)

        # Single capitalized words that appear multiple times (likely important)
        single_pattern = r'\b([A-Z][a-zA-Z]{2,})\b'
        word_counts: Dict[str, int] = {}
        for match in re.finditer(single_pattern, text):
            word = match.group(1)
            if word.lower() not in STOP_WORDS:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Only keep single words that appear 2+ times (likely important)
        for word, count in word_counts.items():
            if count >= 2:
                key = word.lower()
                if key not in seen:
                    seen[key] = self._build(word, "ENTITY", source_doc, min(0.5 + count * 0.05, 0.8))

    def _extract_quoted_terms(self, text: str, source_doc: str, seen: dict):
        """Extract terms in quotes — often important domain terms."""
        for pattern in [r'"([^"]{3,50})"', r"'([^']{3,50})'", r'`([^`]{3,50})`']:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name and name.lower() not in STOP_WORDS:
                    key = name.lower()
                    if key not in seen:
                        seen[key] = self._build(name, "CONCEPT", source_doc, 0.70)

    def _extract_definition_patterns(self, text: str, source_doc: str, seen: dict):
        """
        Extract from definition patterns like:
        'X is a ...', 'X refers to ...', 'The X system ...'
        These are almost always important entities.
        """
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
        """Find known tech terms actually present in the document text."""
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
        words = name.lower().split()
        # Reject if ALL words are stop words
        if all(w in STOP_WORDS for w in words):
            return False
        # Reject pure numbers
        if name.replace(".", "").replace(",", "").isdigit():
            return False
        # Reject very long phrases (probably sentences, not entities)
        if len(name.split()) > 8:
            return False
        return True

    @staticmethod
    def _build(name: str, etype: str, source: str, confidence: float) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "name": name,
            "type": etype,
            "source": source,
            "confidence": round(confidence, 3),
        }
