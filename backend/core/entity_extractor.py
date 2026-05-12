# backend/core/entity_extractor.py

import re
from typing import List, Dict


# Domain-specific tech terms that generic NLP models often miss
TECH_TERMS = {
    "AWS", "Azure", "GCP", "Kubernetes", "Docker", "DevOps",
    "Terraform", "Ansible", "Jenkins", "Kafka", "Spark", "Hadoop",
    "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "FastAPI",
    "LangChain", "Neo4j", "NetworkX", "OpenAI", "Hugging Face",
}


class EntityExtractor:
    """
    Extracts named entities using spaCy NER + a tech-term dictionary overlay.
    Falls back to dictionary-only if spaCy is unavailable.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load(model)
        except Exception as e:
            print(f"[EntityExtractor] spaCy not available ({e}). Using dictionary fallback.")

    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Returns a list of dicts: {"text": ..., "label": ...}
        Labels: PERSON, ORG, GPE, PRODUCT, TECH (custom), etc.
        """
        entities: Dict[str, Dict] = {}  # deduplicate by normalized text

        # --- spaCy NER ---
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Keep only useful entity types
                if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "NORP"}:
                    key = ent.text.strip().lower()
                    if key not in entities:
                        entities[key] = {"text": ent.text.strip(), "label": ent.label_}

        # --- Tech-term dictionary overlay ---
        for term in TECH_TERMS:
            # Case-insensitive whole-word match
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                key = term.lower()
                if key not in entities:
                    entities[key] = {"text": term, "label": "TECH"}

        return list(entities.values())