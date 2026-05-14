"""
RelationshipExtractor — Universal relationship extraction from any document.

Two modes:
  1. Pattern-based — fast regex on sentence text (works for any domain)
  2. Dependency-based — uses spaCy dep-parse to find subject-verb-object triples
     (more accurate, catches relationships not in predefined patterns)
"""
import re
import uuid
from itertools import permutations
from typing import List, Dict, Optional

# Relationship patterns — domain-agnostic ordering matters (specific first)
RELATIONSHIP_PATTERNS = [
    # Tech / infra
    (r'\bintegrates?\s+with\b',      "INTEGRATES_WITH"),
    (r'\bworks?\s+with\b',           "WORKS_WITH"),
    (r'\bdeploys?\s+on\b',           "DEPLOYS_ON"),
    (r'\bruns?\s+on\b',              "RUNS_ON"),
    (r'\borchestrates?\b',           "ORCHESTRATES"),
    (r'\bbuilt\s+(?:on|with)\b',     "BUILT_WITH"),
    (r'\bpowered\s+by\b',            "POWERED_BY"),
    (r'\bconnects?\s+to\b',          "CONNECTS_TO"),
    (r'\bprovides?\b',               "PROVIDES"),
    (r'\benables?\b',                "ENABLES"),
    # General semantic
    (r'\buses?\b',                   "USES"),
    (r'\bsupports?\b',               "SUPPORTS"),
    (r'\bmanages?\b',                "MANAGES"),
    (r'\bcontrols?\b',               "CONTROLS"),
    (r'\bfounds?\b',                 "FOUNDED"),
    (r'\bfounded\s+by\b',            "FOUNDED_BY"),
    (r'\bowned\s+by\b',              "OWNED_BY"),
    (r'\bpartners?\s+with\b',        "PARTNERS_WITH"),
    (r'\bacquired?\b',               "ACQUIRED"),
    (r'\bcompetes?\s+with\b',        "COMPETES_WITH"),
    (r'\bcreated?\s+by\b',           "CREATED_BY"),
    (r'\bdeveloped?\s+by\b',         "DEVELOPED_BY"),
    (r'\bemployed?\s+by\b',          "EMPLOYED_BY"),
    (r'\bworks?\s+(?:at|for)\b',     "WORKS_AT"),
    (r'\blocated?\s+in\b',           "LOCATED_IN"),
    (r'\bbased\s+in\b',              "BASED_IN"),
    (r'\bpart\s+of\b',               "PART_OF"),
    (r'\bsubsidiary\s+of\b',         "SUBSIDIARY_OF"),
    (r'\bdepends?\s+on\b',           "DEPENDS_ON"),
    (r'\brelated\s+to\b',            "RELATED_TO"),
    (r'\bassociated\s+with\b',       "ASSOCIATED_WITH"),
    (r'\bleads?\b',                  "LEADS"),
    (r'\bauthored?\s+by\b',          "AUTHORED_BY"),
    (r'\bpublished?\s+by\b',         "PUBLISHED_BY"),
    (r'\bregulates?\b',              "REGULATES"),
    (r'\bapplied?\s+to\b',           "APPLIED_TO"),
    (r'\btreats?\b',                 "TREATS"),
    (r'\bcauses?\b',                 "CAUSES"),
]


class RelationshipExtractor:
    """
    Universal relationship extractor.
    Works on any document — not just tech content.
    """

    def __init__(self):
        self.nlp = None
        self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            pass  # Will fall back to pattern-only mode

    def extract(self, text: str, entities: List[Dict], source_doc: str = "unknown") -> List[Dict]:
        if not entities:
            return []

        sentences = self._split_sentences(text)
        relationships = []
        seen_keys = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find entities present in this sentence
            present = [
                e for e in entities
                if re.search(r"\b" + re.escape(e["name"]) + r"\b", sentence, re.IGNORECASE)
            ]

            if len(present) < 2:
                continue

            rel_type = self._detect_relationship_type(sentence)

            # Use dependency parsing if available for directional relationships
            if self.nlp and rel_type == "CO_OCCURS_WITH":
                dep_rels = self._dependency_relationships(sentence, present, source_doc)
                for dr in dep_rels:
                    key = (dr["source"], dr["relationship"], dr["target"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        relationships.append(dr)
                if dep_rels:
                    continue  # Don't also add co-occurrence if dep parse found something

            # Pattern-based: create directed edges based on entity order in sentence
            if len(present) == 2:
                src, tgt = present[0], present[1]
                # Order by position in sentence
                pos0 = sentence.lower().find(present[0]["name"].lower())
                pos1 = sentence.lower().find(present[1]["name"].lower())
                if pos1 < pos0:
                    src, tgt = tgt, src

                key = (src["name"], rel_type, tgt["name"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    relationships.append(self._build_rel(
                        src["name"], tgt["name"], rel_type, sentence, source_doc
                    ))
            else:
                # Multiple entities — use all ordered permutations
                for src, tgt in permutations(present, 2):
                    key = (src["name"], rel_type, tgt["name"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        relationships.append(self._build_rel(
                            src["name"], tgt["name"], rel_type, sentence, source_doc,
                            confidence=0.45 if rel_type == "CO_OCCURS_WITH" else 0.7
                        ))

        return relationships

    # -- Helpers -----------------------------------------------------------

    def _dependency_relationships(self, sentence: str, entities: List[Dict], source_doc: str) -> List[Dict]:
        """
        Use spaCy dependency parse to find subject-verb-object triples.
        Extracts directed relationships like (subject, verb, object).
        """
        results = []
        try:
            doc = self.nlp(sentence)
            entity_spans = {}
            for ent in entities:
                name_lower = ent["name"].lower()
                for token in doc:
                    if token.text.lower() in name_lower or name_lower in token.text.lower():
                        entity_spans[token.i] = ent

            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    verb = token.head
                    # Find object of this verb
                    for child in verb.children:
                        if child.dep_ in ("dobj", "pobj", "attr", "prep"):
                            subj_ent = self._match_entity(token, entities)
                            obj_ent = self._match_entity(child, entities)
                            if subj_ent and obj_ent and subj_ent["name"] != obj_ent["name"]:
                                rel_label = verb.lemma_.upper().replace(" ", "_")
                                results.append(self._build_rel(
                                    subj_ent["name"], obj_ent["name"],
                                    rel_label, sentence, source_doc,
                                    confidence=0.80
                                ))
        except Exception:
            pass
        return results

    def _match_entity(self, token, entities: List[Dict]) -> Optional[Dict]:
        """Find which entity a spaCy token belongs to."""
        for ent in entities:
            if token.text.lower() in ent["name"].lower() or ent["name"].lower() in token.text.lower():
                return ent
        return None

    def _detect_relationship_type(self, sentence: str) -> str:
        s = sentence.lower()
        for pattern, rel in RELATIONSHIP_PATTERNS:
            if re.search(pattern, s):
                return rel
        return "CO_OCCURS_WITH"

    def _split_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries, but also on newlines for structured docs
        parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
        # Further split very long sentences on semicolons / colons
        result = []
        for part in parts:
            if len(part) > 300:
                result.extend(re.split(r'[;:]', part))
            else:
                result.append(part)
        return [p.strip() for p in result if p.strip()]

    @staticmethod
    def _build_rel(source: str, target: str, relationship: str,
                   sentence: str, source_doc: str, confidence: float = 0.75) -> Dict:
        return {
            "id": str(uuid.uuid4()),
            "source": source,
            "target": target,
            "relationship": relationship,
            "sentence": sentence,
            "source_doc": source_doc,
            "confidence": confidence,
        }
