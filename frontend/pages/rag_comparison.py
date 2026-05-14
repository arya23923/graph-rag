"""
RAG Comparison page — side-by-side demonstration of Graph RAG vs Traditional RAG.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DEMO_QUERIES = {
    "🔗 Relationship": [
        "How does Kubernetes relate to Docker and AWS?",
        "What does LangChain integrate with?",
        "What is the relationship between Kafka and Spark?",
        "How does Terraform connect to Kubernetes?",
    ],
    "📖 Factual": [
        "What is FastAPI?",
        "Describe Neo4j",
        "What is Kubernetes used for?",
    ],
    "🧩 Multi-hop": [
        "How does Python connect to Neo4j through LangChain?",
        "What AWS-related tools work with Kubernetes?",
    ],
}


def render():
    st.title("⚖️ Graph RAG vs Traditional RAG")
    st.markdown("""
    Compare how **Graph RAG** (entity relationship traversal) and **Traditional RAG** 
    (vector similarity search) answer the same tech questions differently.
    """)

    _show_architecture()
    st.divider()

    # Query selection
    st.subheader("Try a query")
    cat = st.radio("Category:", list(DEMO_QUERIES.keys()), horizontal=True)
    q_options = DEMO_QUERIES[cat]

    selected_q = None
    cols = st.columns(len(q_options))
    for i, q in enumerate(q_options):
        if cols[i].button(q, use_container_width=True, key=f"cmp_{i}"):
            selected_q = q
            st.session_state.cmp_query = q

    custom = st.text_input("Or enter your own query:", value=st.session_state.get("cmp_query",""))
    query = custom or selected_q or st.session_state.get("cmp_query","")

    if st.button("⚖️ Compare Both Modes", type="primary", use_container_width=True) and query:
        _run_comparison(query)

    _show_when_to_use()


def _run_comparison(query: str):
    with st.spinner("Running both RAG pipelines..."):
        from backend.retrieval.graph_rag import GraphRAG
        from backend.retrieval.traditional_rag import TraditionalRAG
        from backend.visualization.graph_viz import GraphViz
        from backend.database.neo4j_client import Neo4jClient

        graph_result = GraphRAG().retrieve(query)
        trad_result  = TraditionalRAG().retrieve(query)

    st.divider()
    st.subheader(f"Query: *{query}*")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🕸️ Graph RAG")
        st.success(graph_result["answer"])

        with st.expander("How it worked"):
            st.markdown(f"**Seed entities found:** `{', '.join(graph_result['seed_entities'])}`")
            st.markdown(f"**Graph hops traversed:** {sum(p['hops'] for p in graph_result.get('paths_traversed',[]))}")
            for snippet in graph_result.get("knowledge_snippets",[]):
                st.caption(f"📌 {snippet}")
            st.markdown("**Relationships retrieved:**")
            for ctx in graph_result.get("graph_context",[])[:8]:
                st.markdown(f"- `{ctx['source']}` → **{ctx['relationship']}** → `{ctx['target']}`")

        st.markdown("**✅ Advantages:**")
        for adv in graph_result.get("advantages",[]):
            st.markdown(f"- {adv}")

    with col2:
        st.markdown("### 📄 Traditional RAG")
        st.info(trad_result["answer"])

        with st.expander("How it worked"):
            st.markdown("**Method:** TF-IDF vector similarity (cosine)")
            for chunk in trad_result.get("retrieved_chunks",[]):
                st.markdown(f"**Score {chunk['score']:.3f}:** {chunk['text'][:120]}...")

        st.markdown("**⚠️ Limitations:**")
        for lim in trad_result.get("limitations",[]):
            st.markdown(f"- {lim}")

    # Graph visualisation for Graph RAG
    if graph_result["seed_entities"]:
        st.divider()
        st.subheader("🌐 Knowledge graph used by Graph RAG")
        seed = graph_result["seed_entities"][0]
        neo4j = Neo4jClient()
        subgraph = neo4j.get_entity_subgraph(seed, depth=2)
        viz = GraphViz()
        fig = viz.render_matplotlib(subgraph["nodes"], subgraph["edges"],
                                    title=f"Graph context: {seed}")
        st.pyplot(fig)

    # Scorecard
    st.divider()
    st.subheader("📊 Scorecard")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Graph entities found",    len(graph_result.get("seed_entities",[])))
    sc2.metric("Graph hops",              sum(p["hops"] for p in graph_result.get("paths_traversed",[])))
    sc3.metric("Traditional chunks used", len(trad_result.get("retrieved_chunks",[])))
    best = trad_result["retrieved_chunks"][0]["score"] if trad_result.get("retrieved_chunks") else 0
    sc4.metric("Best similarity score",  f"{best:.3f}")


def _show_architecture():
    with st.expander("📐 Architecture overview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🕸️ Graph RAG Pipeline")
            st.markdown("""
1. **Entity extraction** — identify tech entities in query  
2. **Graph traversal** — hop through Neo4j relationships  
3. **Context assembly** — collect connected entity facts  
4. **Answer synthesis** — generate using graph context  
            """)
        with col2:
            st.markdown("#### 📄 Traditional RAG Pipeline")
            st.markdown("""
1. **Query vectorisation** — embed the query  
2. **Similarity search** — cosine distance to corpus  
3. **Chunk retrieval** — top-k most similar chunks  
4. **Answer synthesis** — generate from chunks  
            """)


def _show_when_to_use():
    st.divider()
    st.subheader("🤔 When to use which?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Use Graph RAG when:**")
        st.markdown("""
- Asking *how X relates to Y*
- Multi-hop reasoning needed  
- Entity network matters  
- Domain has rich relationships  
        """)
    with col2:
        st.markdown("**Use Traditional RAG when:**")
        st.markdown("""
- Factual / definitional queries  
- Corpus is document-heavy  
- No structured relationships  
- Simpler, faster retrieval needed  
        """)
