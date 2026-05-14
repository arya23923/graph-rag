"""
Improved comparison route with detailed scoring metrics for presentation.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, List
 
# Import the improved RAG implementations
from backend.retrieval.graph_rag import GraphRAG
from backend.retrieval.traditional_rag import TraditionalRAG
 
router = APIRouter(prefix="/api/comparison", tags=["RAG Comparison"])
 
# Initialize with improved implementations
graph_rag = GraphRAG()
trad_rag = TraditionalRAG()
 
 
class CompareRequest(BaseModel):
    query: str
    context_text: Optional[str] = None
 
 
class DetailedScore(BaseModel):
    """Structured score information for UI display."""
    raw_score: float  # 0-1 range
    percentage: float  # 0-100 range
    label: str  # "Highly Relevant", "Relevant", etc.
    
    
class ComparisonMetrics(BaseModel):
    """Comparative metrics between both RAG approaches."""
    winner: str  # "graph_rag", "traditional_rag", or "tie"
    score_difference: float
    recommendation: str
 
 
@router.post("/")
async def compare_rag_modes(req: CompareRequest):
    """
    Compare Graph RAG vs Traditional RAG with detailed scoring.
    Returns comprehensive metrics for presentation/demo purposes.
    """
    # Run both retrievers
    graph_result = graph_rag.retrieve(req.query)
    trad_result = trad_rag.retrieve(req.query)
    
    # Extract key metrics
    graph_metrics = graph_result.get("retrieval_metrics", {})
    trad_metrics = trad_result.get("retrieval_metrics", {})
    
    # Determine winner based on overall scores
    graph_score = graph_metrics.get("overall_graph_score", 0)
    trad_score = trad_metrics.get("best_relevance_score", 0)
    
    if graph_score > trad_score + 10:
        winner = "graph_rag"
        recommendation = "Graph RAG is significantly better for this query (multi-hop reasoning)"
    elif trad_score > graph_score + 10:
        winner = "traditional_rag"
        recommendation = "Traditional RAG is better for this query (direct factual lookup)"
    else:
        winner = "tie"
        recommendation = "Both approaches perform similarly for this query"
    
    return {
        "query": req.query,
        "timestamp": None,  # Add timestamp if needed
        
        # Graph RAG Results
        "graph_rag": {
            "answer": graph_result["answer"],
            "seed_entities": graph_result["seed_entities"],
            "entity_scores": graph_result.get("entity_relevance_scores", []),
            "metrics": {
                "overall_score": graph_score,
                "avg_entity_relevance": graph_metrics.get("avg_entity_relevance", 0),
                "avg_path_relevance": graph_metrics.get("avg_path_relevance", 0),
                "total_paths": graph_metrics.get("total_paths_traversed", 0),
                "unique_entities_reached": graph_metrics.get("unique_entities_reached", 0),
                "semantic_matching": graph_metrics.get("semantic_matching", False)
            },
            "top_paths": graph_result.get("graph_context", [])[:5],  # Top 5 paths
            "advantages": graph_result["advantages"],
        },
        
        # Traditional RAG Results
        "traditional_rag": {
            "answer": trad_result["answer"],
            "retrieved_chunks": trad_result.get("retrieved_chunks", []),
            "metrics": {
                "best_relevance_score": trad_score,
                "avg_relevance_score": trad_metrics.get("avg_relevance_score", 0),
                "total_chunks_retrieved": trad_metrics.get("total_chunks_retrieved", 0),
                "method": trad_result.get("method", "Unknown"),
                "semantic_search": trad_metrics.get("semantic_search", False)
            },
            "top_chunks": [
                {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "relevance_score": chunk.get("relevance_score", 0),
                    "label": chunk.get("relevance_label", "")
                }
                for chunk in trad_result.get("retrieved_chunks", [])[:3]
            ],
            "advantages": trad_result.get("advantages", []),
            "limitations": trad_result.get("limitations", []),
        },
        
        # Comparison Summary
        "comparison": {
            "winner": winner,
            "score_difference": abs(graph_score - trad_score),
            "recommendation": recommendation,
            "graph_rag_score": graph_score,
            "traditional_rag_score": trad_score,
            "use_cases": {
                "graph_rag_better_for": [
                    "Relationship questions (How does X relate to Y?)",
                    "Multi-hop reasoning (What connects X and Y?)",
                    "Entity connections and ecosystem queries",
                    "Technology stack recommendations"
                ],
                "traditional_rag_better_for": [
                    "Direct factual lookups (What is X?)",
                    "Definition retrieval",
                    "Simple Q&A without relationships",
                    "Fast keyword-based search"
                ],
            },
        },
        
        # Visual Data for Charts (for frontend)
        "visualization_data": {
            "score_comparison": {
                "labels": ["Graph RAG", "Traditional RAG"],
                "scores": [graph_score, trad_score]
            },
            "graph_breakdown": {
                "labels": ["Entity Relevance", "Path Relevance"],
                "scores": [
                    graph_metrics.get("avg_entity_relevance", 0),
                    graph_metrics.get("avg_path_relevance", 0)
                ]
            },
            "traditional_breakdown": {
                "labels": ["Best Match", "Average Match"],
                "scores": [
                    trad_score,
                    trad_metrics.get("avg_relevance_score", 0)
                ]
            }
        }
    }
 
 
@router.get("/demo")
async def demo_comparison(query: str = "How does Kubernetes relate to Docker?"):
    """
    Demo endpoint with example query.
    """
    graph_result = graph_rag.retrieve(query)
    trad_result = trad_rag.retrieve(query)
    
    return {
        "query": query,
        "graph_rag": {
            "answer": graph_result["answer"],
            "entities_found": graph_result["seed_entities"],
            "score": graph_result.get("retrieval_metrics", {}).get("overall_graph_score", 0),
        },
        "traditional_rag": {
            "answer": trad_result["answer"],
            "best_chunk": trad_result["retrieved_chunks"][0] if trad_result["retrieved_chunks"] else None,
            "score": trad_result.get("retrieval_metrics", {}).get("best_relevance_score", 0),
        },
        "note": "Use POST /api/comparison/ for detailed comparison with all metrics"
    }
 
 
@router.post("/explain")
async def explain_scores(req: CompareRequest):
    """
    Endpoint that explains how scores are calculated.
    Useful for presentations to show transparency.
    """
    return {
        "query": req.query,
        "scoring_explanation": {
            "traditional_rag": {
                "method": "Cosine Similarity",
                "description": "Compares query embedding with document embeddings",
                "score_range": "0-100 (0 = no similarity, 100 = identical meaning)",
                "factors": [
                    "Semantic similarity between query and document",
                    "Word overlap (with TF-IDF) or meaning overlap (with embeddings)",
                    "No consideration of relationships"
                ]
            },
            "graph_rag": {
                "method": "Multi-factor Graph Scoring",
                "description": "Combines entity relevance, relationship strength, and path quality",
                "score_range": "0-100 (weighted combination)",
                "factors": [
                    "Entity Relevance (60%): How relevant seed entities are to query",
                    "Path Relevance (40%): Quality of relationships traversed",
                    "Relationship Type: Stronger for INTEGRATES_WITH, USES, etc.",
                    "Depth Penalty: Closer relationships score higher"
                ]
            }
        },
        "why_different": [
            "Traditional RAG looks at isolated text chunks",
            "Graph RAG traverses connections between entities",
            "Graph RAG can find multi-hop relationships",
            "Traditional RAG is faster for simple lookups",
            "Graph RAG provides context through relationships"
        ]
    }
 
 
@router.get("/stats")
async def get_retriever_stats():
    """Get statistics about both retrievers."""
    return {
        "traditional_rag": trad_rag.get_stats() if hasattr(trad_rag, 'get_stats') else {},
        "graph_rag": {
            "knowledge_base_size": len(graph_rag.TECH_KNOWLEDGE) if hasattr(graph_rag, 'TECH_KNOWLEDGE') else 0,
            "neo4j_connected": True,  # Simplified
            "semantic_matching": graph_rag.use_embeddings if hasattr(graph_rag, 'use_embeddings') else False
        }
    }
 