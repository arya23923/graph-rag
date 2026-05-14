"""
Improved Graph RAG with relevance scoring and semantic entity matching.
Adds scoring metrics for graph traversal quality and relationship relevance.
"""
import re
import logging
from typing import List, Dict, Optional
import numpy as np
 
logger = logging.getLogger(__name__)
 
# Try to import sentence transformers for semantic entity matching
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available for semantic entity matching")
 
from backend.database.neo4j_client import Neo4jClient
from backend.core.entity_extractor import EntityExtractor
 
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
    "Azure": {
        "description": "Microsoft's cloud computing platform with IaaS, PaaS, and SaaS services.",
        "type": "TECH",
        "related": {"Kubernetes": "HOSTS", "Terraform": "PROVISIONED_BY", "DevOps": "INTEGRATES_WITH"},
    },
    "DevOps": {
        "description": "Software development methodology combining development and operations practices.",
        "type": "CONCEPT",
        "related": {"Jenkins": "USES", "Kubernetes": "USES", "Terraform": "USES", "Azure": "PRACTICED_ON", "AWS": "PRACTICED_ON"},
    },
}
 
 
class GraphRAG:
    """
    Graph RAG with relevance scoring and semantic entity matching.
    Traverses knowledge graph relationships and scores path relevance.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Graph RAG retriever.
        
        Args:
            model_name: Sentence transformer model for semantic entity matching
        """
        self.neo4j = Neo4jClient()
        self.entity_extractor = EntityExtractor(domain="tech")
        self.model_name = model_name
        self.use_embeddings = EMBEDDINGS_AVAILABLE
        
        if self.use_embeddings:
            logger.info(f"Loading semantic model for entity matching: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            logger.info("Semantic entity matching unavailable - using keyword matching")
 
    def retrieve(self, query: str, depth: int = 2, top_k: int = 5) -> Dict:
        """
        Retrieve information via graph traversal with relevance scoring.
        
        Args:
            query: User query string
            depth: How many hops to traverse in the graph
            top_k: Maximum number of entities to start from
            
        Returns:
            Dictionary with answer, graph context, and relevance scores
        """
        # 1. Extract seed entities with semantic matching
        seed_entities = self._extract_seed_entities_semantic(query)
        
        # 2. Calculate query-entity relevance scores
        entity_scores = self._score_entity_relevance(query, seed_entities)
        
        # 3. Traverse graph for each seed
        graph_context = []
        paths_used = []
        relationship_scores = []
        
        for entity_info in entity_scores[:3]:  # Top 3 most relevant entities
            entity = entity_info["entity"]
            entity_score = entity_info["score"]
            
            # Get related entities from graph
            related = self.neo4j.get_related_entities(entity, depth=depth)
            if not related:
                related = self._knowledge_fallback(entity)
            
            # Score each relationship path
            for rel in related:
                path_score = self._score_relationship_path(query, rel, entity_score)
                rel["path_score"] = path_score
                rel["path_relevance"] = round(path_score * 100, 2)
                relationship_scores.append(path_score)
            
            graph_context.extend(related)
            paths_used.append({
                "seed": entity,
                "entity_relevance": entity_info["relevance_score"],
                "hops": len(related)
            })
        
        # 4. Get knowledge snippets
        knowledge_snippets = self._get_knowledge_snippets(seed_entities)
        
        # 5. Calculate aggregate graph metrics
        graph_metrics = self._calculate_graph_metrics(
            seed_entities, entity_scores, graph_context, relationship_scores
        )
        
        # 6. Synthesize answer with scores
        answer = self._synthesise_with_scores(query, entity_scores, graph_context, knowledge_snippets)
        
        return {
            "mode": "graph_rag",
            "query": query,
            "answer": answer,
            "seed_entities": seed_entities,
            "entity_relevance_scores": entity_scores,
            "graph_context": sorted(graph_context, key=lambda x: x.get("path_score", 0), reverse=True)[:20],
            "knowledge_snippets": knowledge_snippets,
            "paths_traversed": paths_used,
            "retrieval_metrics": graph_metrics,
            "advantages": [
                "Traverses multi-hop entity relationships",
                "Captures how technologies interconnect",
                "Provides relationship-aware context with scoring",
                "Can answer 'What works with X?' type questions",
                "Semantic entity matching (understands related concepts)"
            ],
        }
 
    # ── Semantic Entity Extraction ────────────────────────────────────────────
 
    def _extract_seed_entities_semantic(self, query: str) -> List[str]:
        """Extract entities using both keyword matching and semantic similarity."""
        found = []
        
        # 1. Exact keyword matching (fast path)
        for name in TECH_KNOWLEDGE:
            if re.search(r"\b" + re.escape(name) + r"\b", query, re.IGNORECASE):
                found.append(name)
        
        # 2. Try entity extractor
        if not found:
            entities = self.entity_extractor.extract(query)
            found = [e["name"] for e in entities if e["name"] in TECH_KNOWLEDGE]
        
        # 3. Semantic similarity matching (if embeddings available)
        if self.use_embeddings and not found:
            found = self._semantic_entity_matching(query)
        
        # 4. Generic keyword fallback
        if not found:
            keywords = {"kubernetes","docker","fastapi","neo4j","langchain","aws","kafka","spark","python","terraform","azure","devops"}
            for word in query.lower().split():
                if word in keywords and word.title() not in found and word.upper() not in found:
                    found.append(word.title())
        
        return found or ["Python"]  # default seed
 
    def _semantic_entity_matching(self, query: str) -> List[str]:
        """Find entities semantically similar to the query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        entity_names = list(TECH_KNOWLEDGE.keys())
        entity_descriptions = [
            f"{name} {TECH_KNOWLEDGE[name]['description']}" 
            for name in entity_names
        ]
        
        entity_embeddings = self.model.encode(entity_descriptions, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, entity_embeddings)[0]
        
        # Get entities with similarity > threshold
        matches = []
        for idx, sim in enumerate(similarities):
            if sim > 0.3:  # Threshold for relevance
                matches.append((entity_names[idx], float(sim)))
        
        # Sort by similarity and return top entities
        matches.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in matches[:3]]
 
    # ── Scoring Methods ───────────────────────────────────────────────────────
 
    def _score_entity_relevance(self, query: str, entities: List[str]) -> List[Dict]:
        """Score how relevant each entity is to the query."""
        scored_entities = []
        
        if self.use_embeddings:
            # Semantic scoring
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            for entity in entities:
                entity_text = f"{entity} {TECH_KNOWLEDGE.get(entity, {}).get('description', '')}"
                entity_embedding = self.model.encode(entity_text, convert_to_tensor=True)
                similarity = float(util.cos_sim(query_embedding, entity_embedding)[0][0])
                
                scored_entities.append({
                    "entity": entity,
                    "score": similarity,
                    "relevance_score": round(similarity * 100, 2),
                    "relevance_label": self._get_relevance_label(similarity)
                })
        else:
            # Keyword-based scoring (fallback)
            query_lower = query.lower()
            for entity in entities:
                # Simple presence check
                if entity.lower() in query_lower:
                    score = 0.9
                else:
                    score = 0.5
                
                scored_entities.append({
                    "entity": entity,
                    "score": score,
                    "relevance_score": round(score * 100, 2),
                    "relevance_label": self._get_relevance_label(score)
                })
        
        # Sort by score
        scored_entities.sort(key=lambda x: x["score"], reverse=True)
        return scored_entities
 
    def _score_relationship_path(self, query: str, relationship: Dict, entity_score: float) -> float:
        """
        Score the relevance of a relationship path.
        Combines entity relevance, relationship strength, and query relevance.
        """
        # Base score from source entity
        score = entity_score * 0.5
        
        # Add score based on target entity relevance to query
        target = relationship.get("target", "")
        if target.lower() in query.lower():
            score += 0.3
        else:
            score += 0.1
        
        # Relationship type scoring (some relationships are stronger)
        rel_type = relationship.get("relationship", "")
        strong_rels = ["INTEGRATES_WITH", "BUILT_WITH", "USES", "RUNS_ON"]
        if any(strong in rel_type for strong in strong_rels):
            score += 0.15
        else:
            score += 0.05
        
        # Depth penalty (deeper paths less relevant)
        depth = relationship.get("depth", 1)
        score *= (0.9 ** (depth - 1))
        
        return min(score, 1.0)
 
    def _get_relevance_label(self, score: float) -> str:
        """Convert numerical score to human-readable label."""
        if score >= 0.7:
            return "Highly Relevant"
        elif score >= 0.5:
            return "Relevant"
        elif score >= 0.3:
            return "Somewhat Relevant"
        else:
            return "Low Relevance"
 
    # ── Metrics Calculation ───────────────────────────────────────────────────
 
    def _calculate_graph_metrics(
        self, 
        seeds: List[str], 
        entity_scores: List[Dict],
        graph_context: List[Dict],
        relationship_scores: List[float]
    ) -> Dict:
        """Calculate aggregate metrics for graph retrieval quality."""
        
        avg_entity_score = np.mean([e["score"] for e in entity_scores]) if entity_scores else 0
        avg_relationship_score = np.mean(relationship_scores) if relationship_scores else 0
        
        # Count unique entities reached
        unique_targets = set(r["target"] for r in graph_context if "target" in r)
        
        # Overall graph relevance (weighted average)
        overall_score = (avg_entity_score * 0.6 + avg_relationship_score * 0.4)
        
        return {
            "total_entities_found": len(seeds),
            "total_paths_traversed": len(graph_context),
            "unique_entities_reached": len(unique_targets),
            "avg_entity_relevance": round(avg_entity_score * 100, 2),
            "avg_path_relevance": round(avg_relationship_score * 100, 2),
            "overall_graph_score": round(overall_score * 100, 2),
            "best_entity_score": entity_scores[0]["relevance_score"] if entity_scores else 0,
            "semantic_matching": self.use_embeddings,
            "model": self.model_name if self.use_embeddings else "keyword-based"
        }
 
    # ── Answer Synthesis ──────────────────────────────────────────────────────
 
    def _synthesise_with_scores(
        self, 
        query: str, 
        entity_scores: List[Dict], 
        context: List[Dict], 
        snippets: List[str]
    ) -> str:
        """Generate answer with relevance scores included."""
        if not entity_scores:
            return "Could not identify specific tech entities in your query. Try mentioning technologies by name."
        
        parts = []
        
        # Main entities with scores
        for entity_info in entity_scores[:2]:
            entity = entity_info["entity"]
            score = entity_info["relevance_score"]
            label = entity_info["relevance_label"]
            
            info = TECH_KNOWLEDGE.get(entity, {})
            desc = info.get("description", f"{entity} is a technology in the knowledge graph.")
            
            # Get top related entities
            related_items = [r for r in context if r["source"] == entity]
            related_items.sort(key=lambda x: x.get("path_score", 0), reverse=True)
            
            related_with_scores = []
            for r in related_items[:4]:
                target = r["target"]
                rel_score = r.get("path_relevance", 0)
                related_with_scores.append(f"{target} ({rel_score}/100)")
            
            rel_str = ", ".join(related_with_scores) if related_with_scores else "various technologies"
            
            parts.append(
                f"**{entity}** ({label} - {score}/100): {desc}\n"
                f"Connected to: {rel_str}"
            )
        
        # Relationship between entities if multiple seeds
        if len(entity_scores) > 1:
            seed_names = [e["entity"] for e in entity_scores]
            rels_between = [r for r in context if r["source"] in seed_names and r["target"] in seed_names]
            if rels_between:
                rb = rels_between[0]
                path_score = rb.get("path_relevance", 0)
                parts.append(
                    f"\n**Direct Relationship ({path_score}/100):** "
                    f"{rb['source']} → [{rb['relationship']}] → {rb['target']}"
                )
        
        return "\n\n".join(parts) if parts else "Graph traversal complete but no direct knowledge found."
 
    # ── Helper Methods ────────────────────────────────────────────────────────
 
    def _knowledge_fallback(self, entity: str) -> List[Dict]:
        """Return mock relationships from static knowledge base."""
        info = TECH_KNOWLEDGE.get(entity, {})
        related = info.get("related", {})
        return [
            {
                "source": entity,
                "relationship": rel,
                "target": tgt,
                "target_type": "TECH",
                "depth": 1,
                "path_score": 0.7  # Default score for fallback
            }
            for tgt, rel in related.items()
        ]
 
    def _get_knowledge_snippets(self, entities: List[str]) -> List[str]:
        """Get description snippets for entities."""
        snippets = []
        for name in entities:
            info = TECH_KNOWLEDGE.get(name)
            if info:
                snippets.append(f"{name}: {info['description']}")
        return snippets
 