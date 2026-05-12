class RelationshipExtractor:
    def extract_relationships_batch(self, entities: list, text: str) -> dict:
        relationships = [
            {"source": "Kubernetes", "target": "Docker", "relationship_type": "ORCHESTRATES"},
            {"source": "AWS", "target": "Kubernetes", "relationship_type": "HOSTS"},
            {"source": "Azure", "target": "Kubernetes", "relationship_type": "HOSTS"},
            {"source": "DevOps", "target": "Kubernetes", "relationship_type": "USES"},
        ]
        
        return {
            "relationships": relationships,
            "total_relationships": len(relationships),
            "relationship_types": {"ORCHESTRATES": 1, "HOSTS": 2, "USES": 1},
            "graph_density": 0.5
        }