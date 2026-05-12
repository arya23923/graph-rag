import os
import re
import uuid
from typing import List, Dict

# Domain dictionaries (extend via .env later)
DOMAIN_TERMS = {
    "tech": {
        "AWS", "Azure", "GCP", "Kubernetes", "Docker", "DevOps",
        "Terraform", "Ansible", "Jenkins", "Kafka", "Spark", "Hadoop",
        "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "FastAPI",
        "LangChain", "Neo4j", "NetworkX", "OpenAI", "Hugging Face",
        "Python", "Java", "React", "Node.js"
    }
}

SPACY_USEFUL_LABELS = {
    "PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "NORP"
}

class EntityExtractor:
    """
    Graph-ready Entity Extractor
    spaCy NER + domain dictionary + metadata enrichment
    """

    def __init__(self, model: str = "en_core_web_sm", domain: str = None):
        domain = domain or os.getenv("EXTRACTION_DOMAIN", "tech")
        self.domain_terms = DOMAIN_TERMS.get(domain, DOMAIN_TERMS["tech"])
        self.nlp = None

        try:
            import spacy
            self.nlp = spacy.load(model)
        except Exception as e:
            print(f"[EntityExtractor] spaCy unavailable ({e}). Using dictionary only.")

    # -----------------------------------------------------

    def extract(self, text: str, source_doc: str = "unknown") -> List[Dict]:
        """
        Returns graph-ready entity objects
        """
        seen: Dict[str, Dict] = {}

        # ---------- spaCy NER ----------
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in SPACY_USEFUL_LABELS:
                    key = ent.text.lower().strip()
                    if key not in seen:
                        seen[key] = self._build_entity(
                            text=ent.text.strip(),
                            label=ent.label_,
                            source=source_doc,
                            confidence=0.85,
                        )

        # ---------- Domain dictionary ----------
        for term in self.domain_terms:
            if re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE):
                key = term.lower()
                if key not in seen:
                    seen[key] = self._build_entity(
                        text=term,
                        label="TECH",
                        source=source_doc,
                        confidence=0.95,
                    )

        return list(seen.values())

    # -----------------------------------------------------

    def _build_entity(self, text: str, label: str, source: str, confidence: float):
        return {
            "id": str(uuid.uuid4()),
            "name": text,
            "type": label,
            "source": source,
            "confidence": confidence,
        }