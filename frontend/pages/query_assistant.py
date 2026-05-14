"""
Query Assistant page — relationship-based Q&A over the tech knowledge graph.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

RELATIONSHIP_QUESTIONS = [
    "How does Kubernetes relate to Docker?",
    "What technologies integrate with LangChain?",
    "What runs on AWS in the tech stack?",
    "How is FastAPI connected to Neo4j?",
    "What does Python support in the ecosystem?",
    "How does Kafka integrate with Spark?",
    "What is the relationship between Terraform and Kubernetes?",
    "Which technologies are built with Python?",
]

FACTUAL_QUESTIONS = [
    "What is Kubernetes?",
    "Describe FastAPI",
    "What is Neo4j used for?",
    "Explain LangChain",
]


def render():
    st.title("💬 Query Assistant")
    st.markdown("Ask relationship-based or factual questions about the tech knowledge graph.")

    col1, col2 = st.columns([3, 1])
    with col2:
        mode = st.selectbox("Mode:", ["auto (smart)", "graph", "traditional", "both"])
        depth = st.slider("Graph depth:", 1, 4, 2)
    with col1:
        st.markdown("**Example questions:**")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.caption("🔗 Relationship questions")
            for q in RELATIONSHIP_QUESTIONS[:4]:
                if st.button(q, key=f"rq_{q[:20]}", use_container_width=True):
                    st.session_state.query_input = q
        with ecol2:
            st.caption("📖 Factual questions")
            for q in FACTUAL_QUESTIONS:
                if st.button(q, key=f"fq_{q[:20]}", use_container_width=True):
                    st.session_state.query_input = q

    st.divider()
    query = st.text_input(
        "Your question:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., How does Kubernetes relate to Docker?",
    )

    if st.button("🔍 Ask", use_container_width=True, type="primary") and query.strip():
        _run_query(query, mode, depth)


def _run_query(query: str, mode_label: str, depth: int):
    mode_map = {"auto (smart)": "auto", "graph": "graph", "traditional": "traditional", "both": "both"}
    mode = mode_map.get(mode_label, "auto")

    with st.spinner("Querying knowledge graph..."):
        from backend.retrieval.query_processor import QueryProcessor
        processor = QueryProcessor()
        result = processor.process(query, mode=mode, depth=depth)

    actual_mode = result.get("mode", mode)

    if actual_mode == "comparison":
        _show_comparison(result)
    elif actual_mode == "graph_rag":
        _show_graph_result(result)
    else:
        _show_traditional_result(result)

    # Show graph if graph mode
    if actual_mode in ("graph_rag", "comparison"):
        seed_entities = result.get("seed_entities") or result.get("graph_rag", {}).get("seed_entities", [])
        if seed_entities:
            _show_answer_graph(seed_entities[0], depth)


def _show_graph_result(result: dict):
    st.success("**Graph RAG Answer**")
    st.markdown(result.get("answer", "No answer generated."))

    with st.expander("🔍 Graph context"):
        st.markdown(f"**Seed entities:** {', '.join(result.get('seed_entities', []))}")
        for ctx in result.get("graph_context", [])[:15]:
            st.markdown(f"- `{ctx['source']}` → **{ctx['relationship']}** → `{ctx['target']}`")
        for snippet in result.get("knowledge_snippets", []):
            st.info(snippet)


def _show_traditional_result(result: dict):
    st.info("**Traditional RAG Answer**")
    st.markdown(result.get("answer", "No answer generated."))

    with st.expander("📄 Retrieved chunks"):
        for chunk in result.get("retrieved_chunks", []):
            st.markdown(f"**Score {chunk['score']:.3f}:** {chunk['text']}")


def _show_comparison(result: dict):
    gc = result.get("graph_rag", {})
    tc = result.get("traditional_rag", {})
    c1, c2 = st.columns(2)
    with c1:
        st.success("🕸️ Graph RAG")
        st.markdown(gc.get("answer",""))
    with c2:
        st.info("📄 Traditional RAG")
        st.markdown(tc.get("answer",""))


def _show_answer_graph(entity: str, depth: int):
    with st.expander("🌐 Entity relationship graph"):
        from backend.database.neo4j_client import Neo4jClient
        from backend.visualization.graph_viz import GraphViz
        neo4j = Neo4jClient()
        subgraph = neo4j.get_entity_subgraph(entity, depth)
        viz = GraphViz()
        fig = viz.render_matplotlib(subgraph["nodes"], subgraph["edges"],
                                    title=f"Answer context: {entity}")
        st.pyplot(fig)
