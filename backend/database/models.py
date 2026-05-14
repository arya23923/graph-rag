"""Pydantic models for API request/response bodies."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Entity(BaseModel):
    id: str
    name: str
    type: str
    source: str = "unknown"
    confidence: float = 0.5


class Relationship(BaseModel):
    id: str
    source: str
    target: str
    relationship: str
    sentence: str = ""
    source_doc: str = "unknown"
    confidence: float = 0.5


class GraphData(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


class QueryRequest(BaseModel):
    query: str
    mode: str = "graph"          # "graph" | "traditional" | "both"
    entity: Optional[str] = None
    depth: int = Field(default=2, ge=1, le=5)


class ComparisonRequest(BaseModel):
    query: str
    context_text: Optional[str] = None


class PathRequest(BaseModel):
    source: str
    target: str
    max_depth: int = Field(default=4, ge=1, le=6)
