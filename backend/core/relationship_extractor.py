# backend/core/relationship_extractor.py

import re
from itertools import combinations
from typing import List, Dict


# Verb/keyword patterns that suggest a relationship type
RELATIONSHIP_PATTERNS = [
    (r'\bintegrates?\s+with\b',       "INTEGRATES_WITH"),
    (r'\bworks?\s+with\b',            "WORKS_WITH"),
    (r'\bdeploys?\s+on\b',            "DEPLOYS_ON"),
    (r'\bruns?\s+on\b',               "RUNS_ON"),
    (r'\bmanages?\b',                 "MANAGES"),
    (r'\borchestrates?\b',            "ORCHESTRATES"),
    (r'\bbuilt\s+(?:on|with)\b',      "BUILT_WITH"),
    (r'\bextends?\b',                 "EXTENDS"),
    (r'\bsupports?\b',                "SUPPORTS"),
    (r'\breplaces?\b',                "REPLACES"),
    (r'\bcompetes?\s+with\b',         "COMPETES_WITH"),
    (r'\bcompar(?:es?|ed)\s+to\b',    "COMPARED_TO"),
    (r'\bconnects?\s+to\b',           "CONNECTS_TO"),
    (r'\bpart\s+of\b',                "PART_OF"),
    (r'\buse[sd]?\b',                 "USES"),
]


class RelationshipExtractor:
    """
    Extracts relationships between entities found in the same sentence.

    Strategy:
    1. Split text into sentences.
    2. For each sentence, find which extracted entities appear in it.
    3. If 2+ entities co-occur, look for a typed relationship keyword.
       Fall back to generic CO_OCCURS_WITH if none is found.
    """

    def extract(
        self,
        text: str,
        entities: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Args:
            text:     Full document text.
            entities: Output of EntityExtractor.extract() —
                      list of {"text": ..., "label": ...}

        Returns:
            List of {"source": ..., "target": ..., "relationship": ...}
        """
        if not entities or not text:
            return []

        entity_names = [e["text"] for e in entities]
        sentences = self._split_sentences(text)
        relationships: Dict[tuple, Dict] = {}  # deduplicate

        for sentence in sentences:
            # Find which entities appear in this sentence
            present = [
                name for name in entity_names
                if re.search(r'\b' + re.escape(name) + r'\b', sentence, re.IGNORECASE)
            ]

            if len(present) < 2:
                continue

            # Detect relationship type from sentence keywords
            rel_type = self._detect_relationship_type(sentence)

            # Create a relationship for every pair in this sentence
            for src, tgt in combinations(present, 2):
                key = (src.lower(), tgt.lower())
                if key not in relationships:
                    relationships[key] = {
                        "source": src,
                        "target": tgt,
                        "relationship": rel_type,
                    }

        return list(relationships.values())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter — avoids spaCy dependency here."""
        # Split on ., !, ? followed by whitespace/end
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if len(s.strip()) > 10]

    def _detect_relationship_type(self, sentence: str) -> str:
        """Match sentence against known relationship patterns."""
        lower = sentence.lower()
        for pattern, rel_type in RELATIONSHIP_PATTERNS:
            if re.search(pattern, lower):
                return rel_type
        return "CO_OCCURS_WITH"