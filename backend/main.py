"""
FastAPI entry point — Tech Domain Graph RAG Knowledge Assistant.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.config import config
from backend.api.routes import upload, graph, query, comparison

app = FastAPI(
    title="Tech Graph RAG Knowledge Assistant",
    version="2.0.0",
    description="Graph-based RAG for the tech domain — Neo4j + NetworkX + FastAPI",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(graph.router)
app.include_router(query.router)
app.include_router(comparison.router)


@app.get("/")
async def root():
    return {
        "message": "Tech Graph RAG Knowledge Assistant API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "/docs":                  "Swagger UI",
            "/api/upload/":           "Upload tech documents",
            "/api/graph/build":       "Build graph from text (POST)",
            "/api/graph/entity/{e}":  "Get entity subgraph",
            "/api/graph/path":        "Find path between entities (POST)",
            "/api/graph/stats":       "Graph statistics",
            "/api/query/":            "Query assistant (POST)",
            "/api/comparison/":       "Graph RAG vs Traditional RAG (POST)",
        },
    }


@app.get("/health")
async def health():
    from backend.database.neo4j_client import Neo4jClient
    neo4j = Neo4jClient()
    stats = neo4j.get_graph_stats()
    neo4j.close()
    return {"status": "ok", "neo4j": stats}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
