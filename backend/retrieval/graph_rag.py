"""
Graph RAG — LangChain LCEL pipeline + Neo4j graph traversal + Groq LLM synthesis.
"""
import re
import logging
from typing import List, Dict, Generator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

from backend.database.neo4j_client import Neo4jClient
from backend.core.entity_extractor import EntityExtractor
from backend.retrieval.llm_client import _get_groq_llm, _get_openai_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an intelligent knowledge graph assistant.\n"
    "A user has uploaded documents processed into a knowledge graph with entities and relationships.\n"
    "Answer questions using ONLY the context provided. Rules:\n"
    "- Give clear, well-structured answers\n"
    "- Synthesise information — do NOT copy-paste sentences\n"
    "- Explain relationships between entities in plain English\n"
    "- Use bullet points for multiple items\n"
    "- Be specific and concise (3-6 sentences or bullet points)\n"
    "- Never make up information not present in the context\n"
    "{chat_history}"
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  "QUESTION: {question}\n\nKNOWLEDGE GRAPH CONTEXT:\n{context}"),
])


class GraphRAG:
    def __init__(self):
        self.neo4j = Neo4jClient()
        self.entity_extractor = EntityExtractor(domain="tech")
        self._memory: Dict[str, InMemoryChatMessageHistory] = {}

    # ── Memory helpers ─────────────────────────────────────────────────────

    def _get_memory(self, session_id: str = "default") -> InMemoryChatMessageHistory:
        if session_id not in self._memory:
            self._memory[session_id] = InMemoryChatMessageHistory()
        return self._memory[session_id]

    def clear_memory(self, session_id: str = "default"):
        if session_id in self._memory:
            self._memory[session_id].clear()

    def _get_history_str(self, session_id: str) -> str:
        mem = self._get_memory(session_id)
        msgs = mem.messages[-6:]  # last 3 turns
        if not msgs:
            return ""
        return "\n".join(f"{m.type}: {m.content}" for m in msgs)

    # ── Main retrieve ──────────────────────────────────────────────────────

    def retrieve(self, query: str, depth: int = 2,
                 source_doc: str = None, session_id: str = "default") -> Dict:
        seed_entities = self._semantic_entity_search(query, source_doc)

        graph_context, source_sentences, paths_used = [], [], []
        for entity in seed_entities[:4]:
            related = self._traverse_graph(entity, depth, source_doc)
            graph_context.extend(related)
            paths_used.append({"seed": entity, "hops": len(related)})
            source_sentences.extend(self._get_sentences(entity, source_doc))

        source_sentences = self._deduplicate(source_sentences)

        answer, provider = self._generate_answer(
            query, seed_entities, graph_context, source_sentences, session_id
        )

        mem = self._get_memory(session_id)
        mem.add_user_message(query)
        mem.add_ai_message(answer)

        return {
            "mode":             "graph_rag",
            "query":            query,
            "answer":           answer,
            "llm_provider":     provider,
            "seed_entities":    seed_entities,
            "graph_context":    graph_context[:20],
            "source_sentences": source_sentences[:8],
            "paths_traversed":  paths_used,
            "scope":            source_doc or "all documents",
            "advantages": [
                "Semantic search — understands meaning not just keywords",
                "Multi-hop graph traversal — follows entity connections",
                f"LangChain LCEL chain via {provider} — structured synthesis",
                "Conversation memory — remembers previous questions",
                "Source-grounded — answers from your actual documents",
            ],
        }

    def stream_answer(self, query: str, depth: int = 2,
                      source_doc: str = None, session_id: str = "default") -> Generator:
        """Stream tokens for real-time display in Streamlit."""
        seed_entities = self._semantic_entity_search(query, source_doc)
        graph_context, source_sentences = [], []
        for entity in seed_entities[:4]:
            graph_context.extend(self._traverse_graph(entity, depth, source_doc))
            source_sentences.extend(self._get_sentences(entity, source_doc))
        source_sentences = self._deduplicate(source_sentences)

        context_str = self._build_context(seed_entities, graph_context, source_sentences)
        history = self._get_history_str(session_id)
        history_block = f"\nPrevious conversation:\n{history}\n" if history else ""

        llm = _get_groq_llm() or _get_openai_llm()
        if not llm:
            yield self._rule_based_answer(query, seed_entities, graph_context, source_sentences)
            return

        chain = _PROMPT | llm | StrOutputParser()
        full = ""
        for chunk in chain.stream({"question": query, "context": context_str,
                                    "chat_history": history_block}):
            full += chunk
            yield chunk

        mem = self._get_memory(session_id)
        mem.add_user_message(query)
        mem.add_ai_message(full)

    # ── Semantic Entity Search ─────────────────────────────────────────────

    def _semantic_entity_search(self, query: str, source_doc: str = None) -> List[str]:
        found = []
        clean_words = [w.strip("?.,!:;'\"") for w in query.split() if len(w) > 2]
        for word in clean_words:
            for r in self._neo4j_search(word, source_doc, limit=2):
                if r["name"] not in found:
                    found.append(r["name"])

        for term in self._expand_semantically(query):
            for r in self._neo4j_search(term, source_doc, limit=2):
                if r["name"] not in found:
                    found.append(r["name"])

        if not found:
            found = [r["name"] for r in self._neo4j_search(query[:80], source_doc, limit=5)]
        if not found:
            for e in self.entity_extractor.extract(query):
                if self._neo4j_search(e["name"], source_doc, limit=1):
                    found.append(e["name"])
        if not found:
            found = self._get_top_entities(source_doc, limit=3)
        return found[:5]

    def _expand_semantically(self, query: str) -> List[str]:
        q = query.lower()
        intent_map = {
            "how does": ["process", "workflow", "pipeline", "mechanism", "steps"],
            "what is":  ["definition", "overview", "introduction", "concept"],
            "what are": ["types", "categories", "list", "overview"],
            "why":      ["purpose", "reason", "benefit", "goal"],
            "differ":   ["comparison", "contrast", "versus", "difference"],
            "deploy":   ["deployment", "release", "launch", "infrastructure"],
            "train":    ["training", "model", "learning", "algorithm"],
            "data":     ["dataset", "database", "storage", "pipeline"],
        }
        expansions = []
        for keyword, synonyms in intent_map.items():
            if keyword in q:
                expansions.extend(synonyms)
        skip = {"what","how","does","the","are","is","this","that","when","where",
                "which","who","will","can","could","should","have","did","was","were"}
        nouns = [w.strip("?.,!") for w in query.split() if len(w) > 3 and w.lower() not in skip]
        expansions.extend(nouns)
        return list(set(expansions))[:12]

    # ── Neo4j operations ───────────────────────────────────────────────────

    def _neo4j_search(self, query: str, source_doc: str = None, limit: int = 5) -> List[Dict]:
        if not self.neo4j.connected:
            return []
        try:
            with self.neo4j._driver.session() as session:
                cypher = (
                    "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($q)"
                    + (" AND e.source=$doc" if source_doc else "")
                    + " RETURN e.name AS name, e.type AS type, e.confidence AS confidence"
                    + " ORDER BY e.confidence DESC LIMIT $limit"
                )
                params = {"q": query, "limit": limit}
                if source_doc:
                    params["doc"] = source_doc
                return [dict(r) for r in session.run(cypher, **params)]
        except Exception as e:
            logger.debug("Search failed: %s", e)
            return []

    def _traverse_graph(self, entity: str, depth: int, source_doc: str = None) -> List[Dict]:
        if not self.neo4j.connected:
            return []
        try:
            with self.neo4j._driver.session() as session:
                doc_filter = "WHERE start.source=$doc OR related.source=$doc " if source_doc else ""
                result = session.run(
                    f"MATCH path=(start:Entity {{name:$name}})-[*1..{depth}]-(related:Entity) "
                    f"{doc_filter}"
                    "WITH related,[r IN relationships(path)|type(r)] AS rels,length(path) AS hop "
                    "RETURN nodes(path)[-2].name AS source, rels[-1] AS relationship, "
                    "related.name AS target, related.type AS target_type, hop "
                    "ORDER BY hop LIMIT 60",
                    **{"name": entity, **({"doc": source_doc} if source_doc else {})}
                )
                return [{"source": r["source"], "relationship": r["relationship"],
                         "target": r["target"], "target_type": r["target_type"],
                         "depth": r["hop"]} for r in result]
        except Exception as e:
            logger.debug("Traversal failed: %s", e)
            return []

    def _get_sentences(self, entity: str, source_doc: str = None) -> List[Dict]:
        if not self.neo4j.connected:
            return []
        try:
            with self.neo4j._driver.session() as session:
                cypher = (
                    "MATCH (a:Entity {name:$name})-[r]->(b:Entity) "
                    "WHERE r.sentence IS NOT NULL AND r.sentence <> ''"
                    + (" AND r.source_doc=$doc" if source_doc else "")
                    + " RETURN a.name AS source, type(r) AS relationship, b.name AS target,"
                    + " r.sentence AS sentence, r.source_doc AS doc, r.confidence AS confidence"
                    + " ORDER BY r.confidence DESC LIMIT 12"
                )
                params = {"name": entity}
                if source_doc:
                    params["doc"] = source_doc
                return [dict(r) for r in session.run(cypher, **params)]
        except Exception as e:
            logger.debug("Sentence fetch failed: %s", e)
            return []

    def _get_top_entities(self, source_doc: str = None, limit: int = 3) -> List[str]:
        if not self.neo4j.connected:
            return []
        try:
            with self.neo4j._driver.session() as session:
                cypher = (
                    "MATCH (e:Entity)"
                    + (" WHERE e.source=$doc" if source_doc else "")
                    + " RETURN e.name AS name ORDER BY e.confidence DESC LIMIT $limit"
                )
                params = {"limit": limit}
                if source_doc:
                    params["doc"] = source_doc
                return [row["name"] for row in session.run(cypher, **params)]
        except Exception:
            return []

    def _deduplicate(self, sentences: List[Dict]) -> List[Dict]:
        seen, unique = set(), []
        for s in sentences:
            txt = s.get("sentence", "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                unique.append(s)
        return unique

    # ── Answer Generation (LangChain LCEL) ────────────────────────────────

    def _generate_answer(self, query, seeds, context, sentences, session_id) -> tuple:
        if not seeds and not context and not sentences:
            return (
                "❌ No relevant information found.\n\n"
                "**Try:**\n"
                "- Upload a document related to your query\n"
                "- Select the correct document scope\n"
                "- Rephrase using words from your document",
                "none",
            )

        context_str = self._build_context(seeds, context, sentences)
        history = self._get_history_str(session_id)
        history_block = f"\nPrevious conversation:\n{history}\n" if history else ""

        llm = _get_groq_llm() or _get_openai_llm()
        if llm:
            try:
                chain  = _PROMPT | llm | StrOutputParser()
                answer = chain.invoke({
                    "question":     query,
                    "context":      context_str,
                    "chat_history": history_block,
                })
                provider = "Groq (LLaMA 3)" if "groq" in type(llm).__module__ else "OpenAI GPT-3.5"
                return answer.strip(), provider
            except Exception as e:
                logger.warning("LangChain chain failed: %s", e)

        return self._rule_based_answer(query, seeds, context, sentences), "rule-based"

    def _build_context(self, seeds, context, sentences) -> str:
        parts = []
        if seeds:
            parts.append(f"MAIN ENTITIES: {', '.join(seeds)}")
        if sentences:
            parts.append("\nRELEVANT SENTENCES FROM YOUR DOCUMENT:")
            for s in sentences[:10]:
                txt = s.get("sentence", "").strip()
                if txt:
                    parts.append(f"- {txt}")
        if context:
            parts.append("\nKNOWLEDGE GRAPH RELATIONSHIPS:")
            seen = set()
            for r in context[:15]:
                k = (r["source"], r["relationship"], r["target"])
                if k not in seen:
                    seen.add(k)
                    parts.append(f"- {r['source']} --[{r['relationship']}]--> {r['target']}")
        return "\n".join(parts)

    def _rule_based_answer(self, query, seeds, context, sentences) -> str:
        q = query.lower()
        parts = []
        is_how  = any(w in q for w in ["how does", "how do", "how is"])
        is_list = any(w in q for w in ["which", "list", "all", "types"])

        for seed in seeds[:2]:
            outgoing = [(r["relationship"], r["target"]) for r in context if r["source"] == seed]
            if is_how and outgoing:
                parts.append(f"**{seed}** works by:")
                for rel, tgt in outgoing[:5]:
                    parts.append(f"- {rel.replace('_', ' ').lower()} **{tgt}**")
            elif is_list and outgoing:
                parts.append(f"**{seed}** relates to:\n" +
                              "\n".join(f"- {t}" for _, t in outgoing[:8]))
            elif outgoing:
                parts.append(f"**{seed}** → " + " | ".join(
                    f"{r.replace('_', ' ')} **{t}**" for r, t in outgoing[:5]))

        parts.append(
            "\n---\n💡 *Add `GROQ_API_KEY` in `.env` for intelligent LLaMA 3 answers (free!).*\n"
            "Get key at: https://console.groq.com"
        )
        return "\n".join(parts) if len(parts) > 1 else (
            f"Found: **{', '.join(seeds)}** in the graph. "
            "Add Groq API key for detailed intelligent answers."
        )