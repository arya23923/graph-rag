from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.database.neo4j_client import Neo4jClient
from backend.core.graph_builder import GraphBuilder

router = APIRouter(prefix="/api/graph", tags=["Graph Operations"])
neo4j = Neo4jClient()
builder = GraphBuilder()


class BuildRequest(BaseModel):
    text: str
    source_doc: str = "inline"


class PathRequest(BaseModel):
    source: str
    target: str
    max_depth: int = 4


@router.get("/stats")
async def graph_stats():
    return neo4j.get_graph_stats()


@router.get("/entities")
async def list_entities(limit: int = 100):
    return {"entities": neo4j.get_all_entities(limit=limit)}


@router.get("/entity/{entity}")
async def get_entity_graph(entity: str, depth: int = 2):
    try:
        related = neo4j.get_related_entities(entity, depth)
        subgraph = neo4j.get_entity_subgraph(entity, depth)
        return {"entity": entity, "depth": depth, "related": related, "subgraph": subgraph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_entities(q: str, limit: int = 10):
    return {"results": neo4j.search_entities(q, limit)}


@router.post("/path")
async def find_path(req: PathRequest):
    paths = neo4j.find_path(req.source, req.target, req.max_depth)
    return {"source": req.source, "target": req.target, "paths": paths}


@router.post("/build")
async def build_graph_from_text(req: BuildRequest):
    try:
        result = builder.build_from_text(req.text, source_doc=req.source_doc)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_graph():
    return neo4j.clear_graph()


class BuildFileRequest(BaseModel):
    filename: str


@router.post("/build-file")
async def build_graph_from_file(req: BuildFileRequest):
    """Build graph from a previously uploaded file (reads actual file content)."""
    from backend.config import config
    file_path = config.UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.filename}")
    try:
        result = builder.build_from_file(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
