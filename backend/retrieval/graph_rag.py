"""
Graph RAG — retrieval via Neo4j entity graph traversal.
Finds seed entities in the query, traverses relationships, builds rich context.
"""
import re
import logging
from typing import List, Dict, Optional
from backend.database.neo4j_client import Neo4jClient
from backend.core.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

# Static tech knowledge used when Neo4j has no data yet
TECH_KNOWLEDGE = {
    "Kubernetes": {
        "description": "Container orchestration platform for automating deployment, scaling, and management.",
        "type": "TECH",
        "related": {"Docker": "ORCHESTRATES", "AWS": "RUNS_ON", "Terraform": "INTEGRATES_WITH", "Jenkins": "INTEGRATES_WITH"},
    },
    "Docker": {
        "description": "OS-level virtualisation to deliver software in portable containers.",
        "type": "TECH",
        "related": {"Kubernetes": "ORCHESTRATED_BY", "AWS": "RUNS_ON", "Python": "BUILT_WITH"},
    },
    "FastAPI": {
        "description": "Modern Python web framework for building high-performance APIs.",
        "type": "TECH",
        "related": {"Python": "BUILT_WITH", "Neo4j": "CONNECTS_TO", "Pydantic": "USES"},
    },
    "Neo4j": {
        "description": "Native graph database storing data as nodes and relationships.",
        "type": "TECH",
        "related": {"LangChain": "INTEGRATES_WITH", "Python": "SUPPORTS", "FastAPI": "CONNECTED_FROM"},
    },
    "LangChain": {
        "description": "Framework for LLM-powered apps with chains, agents, and retrievers.",
        "type": "TECH",
        "related": {"OpenAI": "INTEGRATES_WITH", "Neo4j": "USES", "Python": "BUILT_WITH"},
    },
    "AWS": {
        "description": "Amazon Web Services — cloud platform providing compute, storage, and managed services.",
        "type": "TECH",
        "related": {"Kubernetes": "HOSTS", "Terraform": "PROVISIONED_BY", "Docker": "RUNS"},
    },
    "Kafka": {
        "description": "Distributed event-streaming platform for high-throughput data pipelines.",
        "type": "TECH",
        "related": {"Spark": "INTEGRATES_WITH", "Elasticsearch": "CONNECTS_TO", "AWS": "RUNS_ON"},
    },
    "Spark": {
        "description": "Unified analytics engine for large-scale data processing and ML.",
        "type": "TECH",
        "related": {"Kafka": "INTEGRATES_WITH", "AWS": "RUNS_ON", "Python": "USES"},
    },
    "Python": {
        "description": "High-level general-purpose language dominant in AI, data science, and backend dev.",
        "type": "TECH",
        "related": {"FastAPI": "POWERS", "LangChain": "POWERS", "NetworkX": "SUPPORTS", "Spark": "USED_BY"},
    },
    "Terraform": {
        "description": "Infrastructure-as-code tool for declarative cloud resource provisioning.",
        "type": "TECH",
        "related": {"AWS": "PROVISIONS", "Kubernetes": "DEPLOYS", "Ansible": "WORKS_WITH"},
    },
}


class GraphRAG:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.entity_extractor = EntityExtractor(domain="tech")

    def retrieve(self, query: str, depth: int = 2, top_k: int = 5) -> Dict:
        # 1. Identify seed entities from query
        seed_entities = self._extract_seed_entities(query)

        # 2. Traverse graph for each seed
        graph_context = []
        paths_used = []
        for entity in seed_entities[:3]:
            related = self.neo4j.get_related_entities(entity, depth=depth)
            if not related:
                related = self._knowledge_fallback(entity)
            graph_context.extend(related)
            paths_used.append({"seed": entity, "hops": len(related)})

        # 3. Also pull static knowledge
        knowledge_snippets = self._get_knowledge_snippets(seed_entities)

        # 4. Synthesise answer
        answer = self._synthesise(query, seed_entities, graph_context, knowledge_snippets)

        return {
            "mode": "graph_rag",
            "query": query,
            "answer": answer,
            "seed_entities": seed_entities,
            "graph_context": graph_context[:20],
            "knowledge_snippets": knowledge_snippets,
            "paths_traversed": paths_used,
            "advantages": [
                "Traverses multi-hop entity relationships",
                "Captures how technologies interconnect",
                "Provides relationship-aware context",
                "Can answer 'What works with X?' type questions",
            ],
        }

    # ── helpers ────────────────────────────────────────────────────────────

    def _extract_seed_entities(self, query: str) -> List[str]:
        found = []
        for name in TECH_KNOWLEDGE:
            if re.search(r"\b" + re.escape(name) + r"\b", query, re.IGNORECASE):
                found.append(name)

        # Also try entity extractor
        if not found:
            entities = self.entity_extractor.extract(query)
            found = [e["name"] for e in entities if e["name"] in TECH_KNOWLEDGE]

        # Generic keyword fallback
        keywords = {"kubernetes","docker","fastapi","neo4j","langchain","aws","kafka","spark","python","terraform"}
        for word in query.lower().split():
            if word in keywords and word.title() not in found and word.upper() not in found:
                found.append(word.title())

        return found or ["Python"]  # default seed

    def _knowledge_fallback(self, entity: str) -> List[Dict]:
        """Return mock relationships from static knowledge base."""
        info = TECH_KNOWLEDGE.get(entity, {})
        related = info.get("related", {})
        return [
            {"source": entity, "relationship": rel, "target": tgt, "target_type": "TECH", "depth": 1}
            for tgt, rel in related.items()
        ]

    def _get_knowledge_snippets(self, entities: List[str]) -> List[str]:
        snippets = []
        for name in entities:
            info = TECH_KNOWLEDGE.get(name)
            if info:
                snippets.append(f"{name}: {info['description']}")
        return snippets

    def _synthesise(self, query: str, seeds: List[str], context: List[Dict], snippets: List[str]) -> str:
        if not seeds:
            return "Could not identify specific tech entities in your query. Try mentioning technologies by name."

        parts = []
        for seed in seeds[:2]:
            info = TECH_KNOWLEDGE.get(seed, {})
            desc = info.get("description", f"{seed} is a technology in the knowledge graph.")
            related_names = [r["target"] for r in context if r["source"] == seed][:4]
            rel_str = ", ".join(related_names) if related_names else "various technologies"
            parts.append(f"**{seed}**: {desc} It is connected to: {rel_str}.")

        if len(seeds) > 1:
            rels_between = [r for r in context if r["source"] in seeds and r["target"] in seeds]
            if rels_between:
                rb = rels_between[0]
                parts.append(f"\nRelationship: {rb['source']} → [{rb['relationship']}] → {rb['target']}.")

        return "\n".join(parts) if parts else "Graph traversal complete but no direct knowledge found."
