"""
Traditional RAG — LangChain-powered with semantic embeddings + LLM synthesis.

Key improvements over the original:
  - _synthesise() now calls an LLM (Groq / OpenAI) to produce a real answer
  - Graceful no-LLM fallback: still builds a coherent answer from chunks
  - Corpus loads BOTH relationship sentences AND entity descriptions so
    factual / definitional queries actually find relevant text
  - FAISS score converted correctly (cosine similarity via relevance_scores)
  - top_k increased to 5 for richer context
  - Source deduplication so diverse docs surface, not the same sentence 3×
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── LLM prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions from retrieved document chunks.\n"
    "Rules:\n"
    "- Answer ONLY from the provided context chunks — never invent information\n"
    "- Give a clear, direct, well-structured answer (3-6 sentences or bullet points)\n"
    "- Synthesise information across chunks — do NOT copy-paste them verbatim\n"
    "- If the context is insufficient to answer fully, say so briefly\n"
    "- Be specific and concise\n"
)

_USER_TEMPLATE = (
    "QUESTION: {query}\n\n"
    "RETRIEVED CHUNKS (most relevant first):\n"
    "{chunks}\n\n"
    "Answer the question based on the chunks above."
)


class TraditionalRAG:
    def __init__(self):
        self._corpus: List[Dict] = []
        self._vectorstore = None
        self._tfidf_matrix = None
        self._tfidf_vectorizer = None
        self._use_embeddings = self._try_load_embeddings()

    # ── Embedding setup ────────────────────────────────────────────────────

    def _try_load_embeddings(self) -> bool:
        """Try to set up HuggingFace embeddings (free, local). Falls back to TF-IDF."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},   # cosine-ready
            )
            logger.info("TraditionalRAG: HuggingFace embeddings loaded ✓")
            return True
        except Exception as e:
            logger.info("TraditionalRAG: embeddings unavailable (%s), using TF-IDF", e)
            self._embeddings = None
            return False

    # ── Public API ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5, source_doc: str = None) -> Dict:
        self._load_corpus(source_doc)
        if not self._corpus:
            return self._empty_result(query, source_doc)

        if self._use_embeddings and self._vectorstore:
            chunks = self._semantic_search(query, top_k)
            method = "Semantic (sentence-transformers/all-MiniLM-L6-v2)"
        else:
            chunks = self._tfidf_search(query, top_k)
            method = "TF-IDF cosine similarity"

        # Deduplicate: at most 2 chunks per source doc so diverse docs surface
        chunks = self._deduplicate(chunks, max_per_doc=2)

        answer = self._synthesise(query, chunks, method)
        return {
            "mode":             "traditional_rag",
            "query":            query,
            "answer":           answer,
            "retrieved_chunks": chunks,
            "corpus_size":      len(self._corpus),
            "scope":            source_doc or "all documents",
            "method":           method,
            "limitations": [
                "Retrieves isolated sentences — no relationship traversal",
                "Cannot answer multi-hop questions (X → Y → Z)",
                "No understanding of how entities connect",
            ],
        }

    # ── Corpus loading ─────────────────────────────────────────────────────

    def _load_corpus(self, source_doc: str = None):
        """
        Build corpus from two Neo4j sources so both factual and relational
        queries find relevant text:
          1. Relationship sentences  (r.sentence on edges)
          2. Entity descriptions     (n.description on nodes)
        """
        try:
            from backend.database.neo4j_client import Neo4jClient
            neo4j = Neo4jClient()
            if not neo4j.connected:
                self._corpus = []
                return

            rows: List[tuple] = []
            with neo4j._driver.session() as session:
                # ── Source 1: relationship sentences ──────────────────────
                doc_filter = " AND r.source_doc = $doc" if source_doc else ""
                rel_cypher = (
                    "MATCH ()-[r]->() WHERE r.sentence IS NOT NULL AND r.sentence <> ''"
                    + doc_filter
                    + " RETURN r.sentence AS text, r.source_doc AS doc"
                    + " ORDER BY r.confidence DESC LIMIT 400"
                )
                params = {"doc": source_doc} if source_doc else {}
                for rec in session.run(rel_cypher, **params):
                    rows.append((rec["text"], rec["doc"] or "unknown"))

                # ── Source 2: entity descriptions ──────────────────────────
                ent_filter = " AND n.source_doc = $doc" if source_doc else ""
                ent_cypher = (
                    "MATCH (n) WHERE n.description IS NOT NULL AND n.description <> ''"
                    + ent_filter
                    + " RETURN n.description AS text, n.source_doc AS doc LIMIT 200"
                )
                for rec in session.run(ent_cypher, **params):
                    rows.append((rec["text"], rec["doc"] or "unknown"))

            if not rows:
                self._corpus = []
                return

            # Deduplicate identical sentences
            seen: set = set()
            unique: List[Dict] = []
            for text, doc in rows:
                if text not in seen:
                    seen.add(text)
                    unique.append({"text": text, "doc": doc})
            self._corpus = unique

            texts = [c["text"] for c in self._corpus]
            if self._use_embeddings:
                self._build_vectorstore(texts)
            else:
                self._build_tfidf(texts)

        except Exception as e:
            logger.warning("Corpus load failed: %s", e)
            self._corpus = []

    def _build_vectorstore(self, texts: List[str]):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
            docs = [Document(page_content=t, metadata={"idx": i})
                    for i, t in enumerate(texts)]
            self._vectorstore = FAISS.from_documents(docs, self._embeddings)
        except Exception as e:
            logger.warning("FAISS build failed (%s), falling back to TF-IDF", e)
            self._use_embeddings = False
            self._build_tfidf(texts)

    def _build_tfidf(self, texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """
        similarity_search_with_relevance_scores returns scores in [0, 1]
        (cosine similarity when embeddings are normalised), not raw L2 distance.
        """
        try:
            results = self._vectorstore.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            chunks = []
            for doc, score in results:
                idx = doc.metadata.get("idx", 0)
                if idx < len(self._corpus):
                    chunks.append({
                        "text":  doc.page_content,
                        "doc":   self._corpus[idx]["doc"],
                        "score": round(float(score), 3),
                    })
            return chunks
        except Exception as e:
            logger.warning("Semantic search failed: %s — falling back to TF-IDF", e)
            return self._tfidf_search(query, top_k)

    def _tfidf_search(self, query: str, top_k: int) -> List[Dict]:
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec  = self._tfidf_vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
            top_idx = scores.argsort()[::-1][:top_k]
            return [
                {
                    "text":  self._corpus[i]["text"],
                    "doc":   self._corpus[i]["doc"],
                    "score": round(float(scores[i]), 3),
                }
                for i in top_idx if scores[i] > 0.01
            ]
        except Exception as e:
            logger.warning("TF-IDF search failed: %s", e)
            return []

    # ── De-duplication ─────────────────────────────────────────────────────

    def _deduplicate(self, chunks: List[Dict], max_per_doc: int = 2) -> List[Dict]:
        """Keep at most `max_per_doc` chunks from the same source document."""
        seen_docs: Dict[str, int] = {}
        result: List[Dict] = []
        for c in chunks:
            doc = c.get("doc", "unknown")
            count = seen_docs.get(doc, 0)
            if count < max_per_doc:
                result.append(c)
                seen_docs[doc] = count + 1
        return result

    # ── Synthesis ──────────────────────────────────────────────────────────

    def _synthesise(self, query: str, chunks: List[Dict], method: str) -> str:
        """
        1. If an LLM is available → ask it to synthesise a real answer.
        2. Otherwise → build a readable summary from the chunks directly.
        """
        if not chunks:
            return (
                "No relevant text found in the selected document(s).\n\n"
                "Try uploading a document that contains information about this topic."
            )

        # Try LLM synthesis first
        llm_answer = self._llm_synthesise(query, chunks)
        if llm_answer:
            return llm_answer

        # Fallback: structured plain-text summary (no LLM configured)
        return self._fallback_answer(query, chunks, method)

    def _llm_synthesise(self, query: str, chunks: List[Dict]) -> Optional[str]:
        """Call Groq / OpenAI to synthesise an answer from the retrieved chunks."""
        try:
            from backend.retrieval.llm_client import get_llm_answer
            chunk_text = "\n".join(
                f"[{i+1}] (source: {c['doc']}, relevance: {c['score']:.2f})\n{c['text']}"
                for i, c in enumerate(chunks)
            )
            user_prompt = _USER_TEMPLATE.format(query=query, chunks=chunk_text)
            result = get_llm_answer(_SYSTEM_PROMPT, user_prompt, max_tokens=500)
            if result["success"] and result["answer"]:
                provider = result.get("provider", "LLM")
                docs_used = list(dict.fromkeys(c["doc"] for c in chunks))  # unique, ordered
                doc_label = docs_used[0] + ("…" if len(docs_used) > 1 else "")
                return (
                    f"{result['answer']}\n\n"
                    f"*Synthesised by {provider} from {len(chunks)} chunks | source: {doc_label}*"
                )
        except Exception as e:
            logger.warning("LLM synthesis failed: %s", e)
        return None

    def _fallback_answer(self, query: str, chunks: List[Dict], method: str) -> str:
        """
        Readable answer when no LLM is configured.
        Groups chunks by document and presents them cleanly.
        """
        lines: List[str] = [
            f"**Relevant passages** for: *{query}*\n",
            f"*Retrieval method: {method} — no LLM configured for synthesis*\n",
        ]
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"**[{i}]** *(relevance: {c['score']:.2f} | `{c['doc']}`)*\n"
                f"{c['text']}\n"
            )
        lines.append(
            "\n💡 *Add `GROQ_API_KEY` to your `.env` (free at console.groq.com) "
            "to get synthesised answers instead of raw passages.*"
        )
        return "\n".join(lines)

    def _empty_result(self, query: str, source_doc: str) -> Dict:
        scope = f"`{source_doc}`" if source_doc else "any document"
        return {
            "mode":             "traditional_rag",
            "query":            query,
            "answer": (
                f"No text found in {scope}.\n\n"
                "Please upload a document first via **Upload & Build Graph**."
            ),
            "retrieved_chunks": [],
            "corpus_size":      0,
            "scope":            source_doc or "all documents",
            "method":           "N/A",
            "limitations":      [],
        }
