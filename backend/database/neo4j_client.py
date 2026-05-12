class Neo4jClient:
    def __init__(self):
        self.connected = False
    
    async def create_knowledge_graph(self, entities: list, relationships: list) -> dict:
        # Mock implementation for testing
        return {
            "entities_created": len(entities),
            "relationships_created": len(relationships),
            "status": "success (mock)"
        }
    
    def get_related_entities(self, entity_name: str, depth: int = 2):
        return []
    
    def close(self):
        pass