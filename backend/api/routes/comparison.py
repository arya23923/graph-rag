from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.retrieval.graph_rag import GraphRAG
from backend.retrieval.traditional_rag import TraditionalRAG

router = APIRouter(prefix="/api/comparison", tags=["RAG Comparison"])
graph_rag = GraphRAG()
trad_rag  = TraditionalRAG()


class CompareRequest(BaseModel):
    query: str
    context_text: Optional[str] = None


@router.post("/")
async def compare_rag_modes(req: CompareRequest):
    graph_result = graph_rag.retrieve(req.query)
    trad_result  = trad_rag.retrieve(req.query)

    return {
        "query": req.query,
        "graph_rag": {
            "answer":          graph_result["answer"],
            "seed_entities":   graph_result["seed_entities"],
            "hops_traversed":  sum(p["hops"] for p in graph_result.get("paths_traversed", [])),
            "advantages":      graph_result["advantages"],
        },
        "traditional_rag": {
            "answer":           trad_result["answer"],
            "chunks_retrieved": len(trad_result["retrieved_chunks"]),
            "best_score":       trad_result["retrieved_chunks"][0]["score"] if trad_result["retrieved_chunks"] else 0,
            "limitations":      trad_result["limitations"],
        },
        "comparison_summary": {
            "graph_rag_better_for":       ["relationship questions", "multi-hop reasoning", "entity connections"],
            "traditional_rag_better_for": ["factual lookups", "exact definition retrieval", "simple Q&A"],
        },
    }


@router.get("/demo")
async def demo_comparison(query: str = "How does Kubernetes relate to Docker?"):
    graph_result = graph_rag.retrieve(query)
    trad_result  = trad_rag.retrieve(query)
    return {
        "query": query,
        "graph_answer": graph_result["answer"],
        "traditional_answer": trad_result["answer"],
        "graph_entities_found": graph_result["seed_entities"],
        "traditional_chunks": [c["text"][:100] for c in trad_result["retrieved_chunks"]],
    }
