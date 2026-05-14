"""
QueryProcessor — routes queries to the right RAG backend and handles relationship questions.
"""
import re
import logging
from typing import Dict, Optional
from backend.retrieval.graph_rag import GraphRAG
from backend.retrieval.traditional_rag import TraditionalRAG

logger = logging.getLogger(__name__)

RELATIONSHIP_PATTERNS = [
    r"how does .+ (?:relate|connect|work) (?:with|to)",
    r"what (?:is|are) .+ (?:connected|related|linked) to",
    r"relationship between .+ and",
    r"how .+ integrates? with",
    r"what (?:uses?|runs? on|deploys? on|works? with)",
    r"path (?:from|between)",
    r"difference between .+ and",
]


class QueryProcessor:
    def __init__(self):
        self.graph_rag = GraphRAG()
        self.traditional_rag = TraditionalRAG()

    def process(self, query: str, mode: str = "auto", depth: int = 2) -> Dict:
        """Route query and return unified response."""
        if mode == "auto":
            mode = "graph" if self._is_relationship_query(query) else "traditional"

        if mode == "graph":
            return self.graph_rag.retrieve(query, depth=depth)
        elif mode == "traditional":
            return self.traditional_rag.retrieve(query)
        elif mode == "both":
            return self._compare(query, depth)
        else:
            return self.graph_rag.retrieve(query, depth=depth)

    def _is_relationship_query(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in RELATIONSHIP_PATTERNS)

    def _compare(self, query: str, depth: int) -> Dict:
        graph_result = self.graph_rag.retrieve(query, depth=depth)
        trad_result  = self.traditional_rag.retrieve(query)
        return {
            "mode": "comparison",
            "query": query,
            "graph_rag": graph_result,
            "traditional_rag": trad_result,
        }
