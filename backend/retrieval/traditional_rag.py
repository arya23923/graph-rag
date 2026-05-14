"""
Improved Traditional RAG with Semantic Search using Sentence Transformers.
Adds proper embeddings, relevance scores, and better answer synthesis.
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
logger = logging.getLogger(__name__)
 
# Try to import sentence transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
 
# Fallback to TF-IDF if embeddings unavailable
from sklearn.feature_extraction.text import TfidfVectorizer
 
 
TECH_CORPUS = [
    "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerised applications.",
    "Docker provides OS-level virtualisation to deliver software in containers, making apps portable and consistent.",
    "AWS offers on-demand cloud computing services including EC2, S3, RDS, Lambda, and managed Kubernetes via EKS.",
    "FastAPI is a modern Python web framework for building APIs with automatic OpenAPI documentation.",
    "Neo4j is a native graph database that stores data as nodes and relationships, ideal for connected data.",
    "LangChain is a framework for building LLM-powered applications with chains, agents, and memory.",
    "Kafka is a distributed event-streaming platform used for high-throughput, fault-tolerant data pipelines.",
    "Spark is a unified analytics engine for large-scale data processing with built-in ML libraries.",
    "Terraform is an infrastructure-as-code tool that allows you to define cloud resources declaratively.",
    "Redis is an in-memory data store used for caching, message brokering, and real-time analytics.",
    "MongoDB is a NoSQL document database that stores data in flexible, JSON-like documents.",
    "NetworkX is a Python library for creation, manipulation, and analysis of complex network graphs.",
    "OpenAI provides large language models including GPT-4 and embedding models via REST API.",
    "Ansible is an agentless IT automation tool for configuration management and application deployment.",
    "Jenkins is an open-source CI/CD automation server widely used in DevOps pipelines.",
    "Elasticsearch is a distributed search and analytics engine built on Apache Lucene.",
    "Python is a high-level, general-purpose programming language widely used in AI, data science, and web development.",
    "React is a JavaScript library for building user interfaces, maintained by Meta.",
    "Graph RAG enhances retrieval by traversing knowledge graph relationships to find contextually connected entities.",
    "Traditional RAG uses vector embeddings and cosine similarity to retrieve the most semantically similar document chunks.",
]
 
 
class TraditionalRAG:
    """
    Traditional RAG with semantic embeddings.
    Uses sentence-transformers for semantic similarity (preferred) or TF-IDF fallback.
    """
    
    def __init__(self, corpus: Optional[List[str]] = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Traditional RAG retriever.
        
        Args:
            corpus: List of text chunks to search through
            model_name: Sentence transformer model (default: all-MiniLM-L6-v2 - fast & good)
                       Other options: 
                       - "all-mpnet-base-v2" (more accurate, slower)
                       - "paraphrase-MiniLM-L6-v2" (good for paraphrase detection)
        """
        self.corpus = corpus or TECH_CORPUS
        self.model_name = model_name
        self.use_embeddings = EMBEDDINGS_AVAILABLE
        
        if self.use_embeddings:
            logger.info(f"Loading semantic embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            # Pre-compute embeddings for corpus
            self.corpus_embeddings = self.model.encode(
                self.corpus, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            logger.info(f"✓ Semantic embeddings computed for {len(self.corpus)} documents")
        else:
            logger.info("Using TF-IDF fallback (keyword matching)")
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._matrix = self._vectorizer.fit_transform(self.corpus)
 
    def add_documents(self, texts: List[str]):
        """Add new documents to the corpus and recompute embeddings."""
        self.corpus.extend(texts)
        
        if self.use_embeddings:
            # Recompute all embeddings
            self.corpus_embeddings = self.model.encode(
                self.corpus, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
        else:
            self._matrix = self._vectorizer.fit_transform(self.corpus)
 
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve most relevant documents using semantic or TF-IDF similarity.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            
        Returns:
            Dictionary with answer, chunks, scores, and metadata
        """
        if self.use_embeddings:
            return self._retrieve_semantic(query, top_k)
        else:
            return self._retrieve_tfidf(query, top_k)
 
    def _retrieve_semantic(self, query: str, top_k: int) -> Dict:
        """Semantic retrieval using sentence embeddings."""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build chunks with normalized scores (0-100 scale for better presentation)
        chunks = []
        for idx in top_indices:
            raw_score = float(similarities[idx])
            if raw_score > 0.01:  # Filter very low relevance
                chunks.append({
                    "text": self.corpus[idx],
                    "score": raw_score,  # Original 0-1 score
                    "relevance_score": round(raw_score * 100, 2),  # 0-100 scale
                    "chunk_id": int(idx),
                    "relevance_label": self._get_relevance_label(raw_score)
                })
        
        # Calculate aggregate metrics
        avg_score = np.mean([c["score"] for c in chunks]) if chunks else 0
        
        answer = self._synthesise_semantic(query, chunks)
        
        return {
            "mode": "traditional_rag",
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "method": f"Semantic Embeddings ({self.model_name})",
            "retrieval_metrics": {
                "total_chunks_retrieved": len(chunks),
                "best_relevance_score": chunks[0]["relevance_score"] if chunks else 0,
                "avg_relevance_score": round(avg_score * 100, 2),
                "embedding_model": self.model_name,
                "semantic_search": True
            },
            "advantages": [
                "Uses semantic similarity (understands meaning, not just keywords)",
                "Captures paraphrases and related concepts",
                "Better than keyword matching for natural language queries",
                "Works well for factual retrieval and definitions"
            ],
            "limitations": [
                "Retrieves isolated chunks; cannot traverse relationships",
                "Misses multi-hop connections (e.g., Kubernetes → Docker → Python)",
                "Answers limited to explicit text in corpus",
                "No understanding of how entities are interconnected",
            ],
        }
 
    def _retrieve_tfidf(self, query: str, top_k: int) -> Dict:
        """TF-IDF fallback retrieval (keyword-based)."""
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
 
        chunks = []
        for i in top_idx:
            raw_score = float(scores[i])
            if raw_score > 0.01:
                chunks.append({
                    "text": self.corpus[i],
                    "score": raw_score,
                    "relevance_score": round(raw_score * 100, 2),
                    "chunk_id": int(i),
                    "relevance_label": self._get_relevance_label(raw_score)
                })
 
        avg_score = np.mean([c["score"] for c in chunks]) if chunks else 0
        answer = self._synthesise_tfidf(query, chunks)
 
        return {
            "mode": "traditional_rag",
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "method": "TF-IDF (keyword matching - fallback)",
            "retrieval_metrics": {
                "total_chunks_retrieved": len(chunks),
                "best_relevance_score": chunks[0]["relevance_score"] if chunks else 0,
                "avg_relevance_score": round(avg_score * 100, 2),
                "semantic_search": False,
                "warning": "Using keyword matching. Install sentence-transformers for semantic search."
            },
            "advantages": [
                "Fast keyword-based retrieval",
                "No model loading required"
            ],
            "limitations": [
                "⚠️ KEYWORD MATCHING ONLY - not semantic",
                "Cannot understand paraphrases or synonyms",
                "Retrieves isolated chunks; cannot traverse relationships",
                "Misses multi-hop connections",
            ],
        }
 
    def _get_relevance_label(self, score: float) -> str:
        """Convert numerical score to human-readable label."""
        if score >= 0.7:
            return "Highly Relevant"
        elif score >= 0.5:
            return "Relevant"
        elif score >= 0.3:
            return "Somewhat Relevant"
        else:
            return "Low Relevance"
 
    def _synthesise_semantic(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer from semantically retrieved chunks."""
        if not chunks:
            return "No relevant documents found for your query."
        
        best = chunks[0]
        answer_parts = [
            f"**Most Relevant ({best['relevance_score']}/100):** {best['text']}"
        ]
        
        if len(chunks) > 1 and chunks[1]["relevance_score"] > 30:
            second = chunks[1]
            answer_parts.append(f"\n\n**Also Relevant ({second['relevance_score']}/100):** {second['text'][:200]}...")
        
        return "\n".join(answer_parts)
 
    def _synthesise_tfidf(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer from TF-IDF chunks (keyword matching)."""
        if not chunks:
            return "No relevant documents found for your query."
        
        best = chunks[0]
        extra = f" Additionally: {chunks[1]['text'][:120]}..." if len(chunks) > 1 else ""
        return f"⚠️ **Keyword Match ({best['relevance_score']}/100):** {best['text']}{extra}\n\n*Note: Using TF-IDF fallback. Install sentence-transformers for semantic search.*"
 
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            "corpus_size": len(self.corpus),
            "using_embeddings": self.use_embeddings,
            "model": self.model_name if self.use_embeddings else "TF-IDF",
            "embedding_dimension": self.corpus_embeddings.shape[1] if self.use_embeddings else None
        }