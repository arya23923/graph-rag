from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

from backend.config import config
from backend.api.routes import upload, graph, query, comparison
from backend.core.text_extractor import TextExtractor
from backend.core.entity_extractor import EntityExtractor
from backend.core.relationship_extractor import RelationshipExtractor
from backend.database.neo4j_client import Neo4jClient

app = FastAPI(title="Graph RAG Knowledge Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(graph.router)
app.include_router(query.router)
app.include_router(comparison.router)

# Initialize components
text_extractor = TextExtractor()
entity_extractor = EntityExtractor()
relationship_extractor = RelationshipExtractor()
neo4j_client = Neo4jClient()

@app.get("/")
async def root():
    return {
        "message": "Graph RAG Knowledge Assistant API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/docs - Swagger documentation",
            "/api/upload - Upload documents",
            "/api/graph - Graph operations",
            "/api/query - Query assistant"
        ]
    }

@app.post("/api/process-document/{filename}")
async def process_document_complete(filename: str):
    """Complete pipeline: Extract text → Entities → Relationships"""
    
    file_path = config.UPLOAD_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Step 1: Extract text
    extraction = text_extractor.extract(file_path)
    if not extraction["success"]:
        raise HTTPException(status_code=500, detail=extraction["error"])
    
    text = extraction["text"]
    
    # Step 2: Extract entities
    entity_result = entity_extractor.extract_entities(text)
    
    # Step 3: Extract relationships
    relationship_result = relationship_extractor.extract_relationships_batch(
        entity_result["entities"], 
        text
    )
    
    # Step 4: Store in Neo4j
    graph_result = await neo4j_client.create_knowledge_graph(
        entity_result["entities"],
        relationship_result["relationships"]
    )
    
    return {
        "status": "success",
        "document": filename,
        "extraction": {
            "char_count": extraction["char_count"],
            "word_count": extraction["word_count"]
        },
        "entities": entity_result,
        "relationships": relationship_result,
        "graph_storage": graph_result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)