from fastapi import APIRouter, HTTPException
from backend.database.neo4j_client import Neo4jClient

router = APIRouter(prefix="/api/graph", tags=["Graph Operations"])

@router.get("/{entity}")
async def get_entity_graph(entity: str, depth: int = 2):
    """Get graph for an entity"""
    client = Neo4jClient()
    try:
        result = client.get_related_entities(entity, depth)
        return {
            "entity": entity,
            "depth": depth,
            "related_entities": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        client.close()