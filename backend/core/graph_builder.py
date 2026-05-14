"""
GraphBuilder -- orchestrates text -> entities -> relationships -> Neo4j.
Universal pipeline: works with any document type, any domain.
"""
import logging
from typing import Dict, Any
from pathlib import Path

from backend.core.text_extractor import TextExtractor
from backend.core.entity_extractor import EntityExtractor
from backend.core.relationship_extractor import RelationshipExtractor
from backend.database.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self):
        self.text_extractor = TextExtractor()
        # domain=None -> auto-detect; tech terms still boosted but NER covers everything
        self.entity_extractor = EntityExtractor(domain=None)
        self.relationship_extractor = RelationshipExtractor()
        self.neo4j = Neo4jClient()

    def build_from_file(self, file_path) -> Dict[str, Any]:
        path = Path(file_path)
        extraction = self.text_extractor.extract(path)
        if not extraction["success"]:
            return {"success": False, "error": extraction["error"]}
        logger.info("GraphBuilder: extracted %d chars from %s",
                    extraction["metadata"].get("char_count", 0), path.name)
        return self.build_from_text(extraction["text"], source_doc=path.name)

    def build_from_text(self, text: str, source_doc: str = "inline") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"success": False, "error": "Empty text -- nothing to process."}

        entities = self.entity_extractor.extract(text, source_doc=source_doc)
        if not entities:
            return {
                "success": False,
                "error": (
                    "No entities found. Try a document with named people, "
                    "organisations, places, products, or technologies."
                )
            }

        relationships = self.relationship_extractor.extract(text, entities, source_doc=source_doc)
        graph_result = self.neo4j.create_knowledge_graph(entities, relationships)

        logger.info("GraphBuilder: %d entities, %d relationships stored for %s",
                    len(entities), len(relationships), source_doc)

        return {
            "success": True,
            "source_doc": source_doc,
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "entities": entities,
            "relationships": relationships,
            "graph_result": graph_result,
        }

    def get_stats(self) -> Dict:
        return self.neo4j.get_graph_stats()
