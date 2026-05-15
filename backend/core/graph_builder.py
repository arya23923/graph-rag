"""
GraphBuilder — orchestrates document -> entities -> relationships -> Neo4j.

LangChain upgrade:
  - Uses new DocumentProcessor (LangChain loaders + RecursiveCharacterTextSplitter)
  - Processes chunks in parallel batches for large documents
  - Falls back to original TextExtractor if LangChain loaders unavailable
"""
import logging
from typing import Dict, Any
from pathlib import Path

from backend.core.entity_extractor import EntityExtractor
from backend.core.relationship_extractor import RelationshipExtractor
from backend.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self):
        self.entity_extractor = EntityExtractor(domain=None)
        self.relationship_extractor = RelationshipExtractor()
        self.neo4j = Neo4jClient()

    def build_from_file(self, file_path) -> Dict[str, Any]:
        path = Path(file_path)

        # Try LangChain document processor first
        try:
            from backend.core.document_processor import load_and_chunk
            result = load_and_chunk(path)
            if result["success"]:
                logger.info("GraphBuilder: LangChain loader — %d chars, %d chunks from %s",
                            result["metadata"]["char_count"],
                            result["metadata"]["chunk_count"],
                            path.name)
                return self._build_from_processed(result["full_text"],
                                                   result["chunks"],
                                                   source_doc=path.name)
        except Exception as e:
            logger.warning("LangChain processor failed (%s), falling back to TextExtractor", e)

        # Fallback: original TextExtractor
        from backend.core.text_extractor import TextExtractor
        extraction = TextExtractor().extract(path)
        if not extraction["success"]:
            return {"success": False, "error": extraction["error"]}
        logger.info("GraphBuilder: TextExtractor — %d chars from %s",
                    extraction["metadata"].get("char_count", 0), path.name)
        return self.build_from_text(extraction["text"], source_doc=path.name)

    def build_from_text(self, text: str, source_doc: str = "inline") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"success": False, "error": "Empty text — nothing to process."}
        return self._build_from_processed(text, [text], source_doc=source_doc)

    def _build_from_processed(self, full_text: str, chunks: list,
                               source_doc: str) -> Dict[str, Any]:
        """Core pipeline: extract entities from chunks, relationships from full text."""
        all_entities: Dict[str, dict] = {}

        # Extract entities per chunk (more accurate than one huge pass)
        for chunk in chunks:
            for ent in self.entity_extractor.extract(chunk, source_doc=source_doc):
                key = ent["name"].lower()
                # Keep highest confidence version
                if key not in all_entities or ent["confidence"] > all_entities[key]["confidence"]:
                    all_entities[key] = ent

        entities = list(all_entities.values())
        if not entities:
            return {
                "success": False,
                "error": (
                    "No entities found. Try a document with named people, "
                    "organisations, places, products, or technologies."
                ),
            }

        # Extract relationships from full text (needs context across chunks)
        relationships = self.relationship_extractor.extract(full_text, entities,
                                                            source_doc=source_doc)
        graph_result = self.neo4j.create_knowledge_graph(entities, relationships)

        logger.info("GraphBuilder: %d entities, %d relationships stored for %s",
                    len(entities), len(relationships), source_doc)

        return {
            "success":             True,
            "source_doc":          source_doc,
            "entities_count":      len(entities),
            "relationships_count": len(relationships),
            "entities":            entities,
            "relationships":       relationships,
            "graph_result":        graph_result,
        }

    def get_stats(self) -> Dict:
        return self.neo4j.get_graph_stats()
