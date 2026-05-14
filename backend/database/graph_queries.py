"""
Reusable Cypher queries for the Tech Domain Graph RAG knowledge assistant.
All functions return plain Python dicts / lists — no Neo4j Record objects leak out.
"""
from typing import List, Dict, Optional


def build_entity_context_query(entity_name: str, depth: int = 2) -> str:
    """Return Cypher that fetches rich context for Graph-RAG retrieval."""
    return f"""
    MATCH (start:Entity {{name: '{entity_name}'}})
    OPTIONAL MATCH path = (start)-[*1..{depth}]-(related:Entity)
    WITH start,
         collect(DISTINCT {{
             name:         related.name,
             type:         related.type,
             relationship: type(last(relationships(path))),
             depth:        length(path)
         }}) AS neighbors
    RETURN start.name          AS entity,
           start.type          AS entity_type,
           neighbors
    """


def build_relationship_query(source: str, target: str) -> str:
    """Return Cypher to find all relationships between two entities."""
    return f"""
    MATCH (src:Entity {{name: '{source}'}})-[r]-(tgt:Entity {{name: '{target}'}})
    RETURN type(r) AS relationship, r.sentence AS evidence, r.confidence AS confidence
    """


def build_tech_stack_query(technology: str) -> str:
    """Find everything a technology integrates with / runs on."""
    return f"""
    MATCH (t:Entity {{name: '{technology}'}})-[r:INTEGRATES_WITH|RUNS_ON|BUILT_WITH|USES|CONNECTS_TO]-(other:Entity)
    RETURN other.name AS tech, type(r) AS relationship, other.type AS tech_type
    ORDER BY other.name
    """


def build_community_query() -> str:
    """Fetch all nodes + edges for full-graph visualisation."""
    return """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.name AS source, a.type AS source_type,
           type(r) AS relationship,
           b.name AS target, b.type AS target_type
    LIMIT 500
    """
