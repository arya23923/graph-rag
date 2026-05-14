"""
Traditional RAG — pure vector similarity search (TF-IDF fallback, no embeddings required).
In production swap TfidfVectorizer for OpenAI / HuggingFace embeddings.
"""
import re
import logging
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

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
    def __init__(self, corpus: Optional[List[str]] = None):
        self.corpus = corpus or TECH_CORPUS
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self.corpus)

    def add_documents(self, texts: List[str]):
        self.corpus.extend(texts)
        self._matrix = self._vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 3) -> Dict:
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        chunks = [
            {"text": self.corpus[i], "score": float(scores[i]), "chunk_id": i}
            for i in top_idx if scores[i] > 0.01
        ]

        answer = self._synthesise(query, chunks)
        return {
            "mode": "traditional_rag",
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "method": "TF-IDF vector similarity",
            "limitations": [
                "Retrieves isolated chunks; cannot traverse relationships",
                "Misses multi-hop connections (e.g., Kubernetes → Docker → Python)",
                "Answers limited to explicit text in corpus",
                "No understanding of how entities are interconnected",
            ],
        }

    def _synthesise(self, query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant documents found for your query."
        best = chunks[0]["text"]
        extra = f" Additionally: {chunks[1]['text'][:120]}..." if len(chunks) > 1 else ""
        return f"Based on document similarity: {best}{extra}"
