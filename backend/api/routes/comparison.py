from fastapi import APIRouter

router = APIRouter(prefix="/api/comparison", tags=["RAG Comparison"])

@router.get("/")
async def traditional_vs_graph_rag(query: str, type: str = "graph"):
    """Compare traditional RAG vs Graph RAG"""
    if type == "traditional":
        return {
            "type": "traditional",
            "response": "Traditional RAG response based on vector similarity",
            "limitations": "May miss relationship connections"
        }
    else:
        return {
            "type": "graph",
            "response": "Graph RAG response using entity relationships",
            "advantages": "Captures multi-hop relationships"
        }