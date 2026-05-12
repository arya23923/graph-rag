import re
import uuid
from itertools import permutations
from typing import List, Dict

RELATIONSHIP_PATTERNS = [
    (r'\bintegrates?\s+with\b', "INTEGRATES_WITH"),
    (r'\bworks?\s+with\b', "WORKS_WITH"),
    (r'\bdeploys?\s+on\b', "DEPLOYS_ON"),
    (r'\bruns?\s+on\b', "RUNS_ON"),
    (r'\borchestrates?\b', "ORCHESTRATES"),
    (r'\buses?\b', "USES"),
    (r'\bsupports?\b', "SUPPORTS"),
    (r'\bconnects?\s+to\b', "CONNECTS_TO"),
    (r'\bbuilt\s+(?:on|with)\b', "BUILT_WITH"),
]

class RelationshipExtractor:
    """
    Produces graph-ready triples for Neo4j ingestion
    """

    def extract(self, text: str, entities: List[Dict], source_doc: str = "unknown") -> List[Dict]:
        if not entities:
            return []

        entity_names = [e["name"] for e in entities]
        sentences = self._split_sentences(text)
        relationships = []

        for sentence in sentences:
            present = [
                e for e in entities
                if re.search(r'\b' + re.escape(e["name"]) + r'\b', sentence, re.IGNORECASE)
            ]

            if len(present) < 2:
                continue

            rel_type = self._detect_relationship_type(sentence)

            for src, tgt in permutations(present, 2):
                relationships.append({
                    "id": str(uuid.uuid4()),
                    "source": src["name"],
                    "target": tgt["name"],
                    "relationship": rel_type,
                    "sentence": sentence,
                    "source_doc": source_doc,
                    "confidence": 0.75 if rel_type != "CO_OCCURS_WITH" else 0.4
                })

        return relationships

    # -----------------------------------------------------

    def _split_sentences(self, text: str):
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def _detect_relationship_type(self, sentence: str):
        s = sentence.lower()
        for pattern, rel in RELATIONSHIP_PATTERNS:
            if re.search(pattern, s):
                return rel
        return "CO_OCCURS_WITH"