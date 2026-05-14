from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.retrieval.query_processor import QueryProcessor

router = APIRouter(prefix="/api/query", tags=["Query Assistant"])
processor = QueryProcessor()


class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"   # auto | graph | traditional | both
    depth: int = 2


@router.post("/")
async def graph_query(request: QueryRequest):
    try:
        result = processor.process(request.query, mode=request.mode, depth=request.depth)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggest")
async def suggest_mode(query: str):
    suggested = "graph" if processor._is_relationship_query(query) else "traditional"
    return {"query": query, "suggested_mode": suggested}
