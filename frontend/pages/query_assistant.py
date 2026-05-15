"""
Query Assistant — LangChain-powered Graph RAG with streaming + conversation memory.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _get_available_docs() -> list:
    try:
        from backend.database.neo4j_client import Neo4jClient
        neo4j = Neo4jClient()
        if not neo4j.connected:
            return []
        with neo4j._driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.source IS NOT NULL "
                "RETURN DISTINCT n.source AS src, count(n) AS cnt "
                "ORDER BY cnt DESC LIMIT 20"
            )
            return [(r["src"], r["cnt"]) for r in result]
    except Exception:
        return []


def render():
    st.title("💬 Query Assistant")
    st.markdown(
        "Ask intelligent questions — answers come from your uploaded documents "
        "via the knowledge graph. **LangChain** powers retrieval, synthesis, and conversation memory."
    )

    available_docs = _get_available_docs()

    # ── Session ID (for per-user memory) ───────────────────────────────────
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]

    # ── Controls ───────────────────────────────────────────────────────────
    col_mode, col_scope, col_depth = st.columns([1, 2, 1])

    with col_mode:
        rag_mode = st.radio(
            "RAG Mode:",
            ["🕸️ Graph RAG", "📄 Traditional RAG", "⚖️ Both"],
            help="Graph RAG uses entity relationships. Traditional uses semantic similarity (FAISS/TF-IDF).",
        )

    with col_scope:
        st.markdown("**Search Scope:**")
        scope_tabs = st.tabs(["🌐 All Documents", "📄 Specific Document"])
        with scope_tabs[0]:
            st.caption("Query searches across ALL uploaded documents.")
            selected_doc = None
            use_all = True
        with scope_tabs[1]:
            if available_docs:
                doc_options = [f"{doc}  ({cnt} entities)" for doc, cnt in available_docs]
                chosen = st.selectbox("Choose document:", doc_options)
                selected_doc = available_docs[doc_options.index(chosen)][0]
                use_all = False
                st.caption(f"✅ Searching only: `{selected_doc}`")
            else:
                st.warning("No documents uploaded yet.")
                selected_doc = None
                use_all = True

    with col_depth:
        depth = st.slider("Graph depth:", 1, 4, 2,
                          help="Hops to traverse in the knowledge graph")

    active_doc = None if use_all else selected_doc

    # ── Memory controls ────────────────────────────────────────────────────
    mem_col, stream_col = st.columns([3, 1])
    with mem_col:
        st.caption(f"🧠 Conversation memory active (session: `{st.session_state.session_id}`)")
    with stream_col:
        use_streaming = st.toggle("⚡ Stream", value=True,
                                  help="Stream tokens in real-time (Graph RAG only)")
        if st.button("🗑️ Clear memory", use_container_width=True):
            try:
                from backend.retrieval.graph_rag import GraphRAG
                GraphRAG().clear_memory(st.session_state.session_id)
                st.success("Memory cleared")
            except Exception:
                pass

    st.divider()

    # ── Example questions ──────────────────────────────────────────────────
    st.markdown("**💡 Try these questions:**")
    example_cols = st.columns(4)
    examples = [
        "What is the main purpose of this system?",
        "How does the process work?",
        "What technologies are used?",
        "How do the components connect?",
        "What does the system use for ML?",
        "Explain the screening process",
        "What are the key features?",
        "How does ranking work?",
    ]
    for i, ex in enumerate(examples):
        if example_cols[i % 4].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.qa_query = ex

    query = st.text_input(
        "Your question:",
        value=st.session_state.get("qa_query", ""),
        placeholder="Ask anything about your uploaded documents...",
    )

    if st.button("🔍 Get Answer", use_container_width=True, type="primary") and query.strip():
        st.session_state.qa_query = ""
        mode_key = rag_mode.split()[1].lower()
        _run_query(query, mode_key, depth, active_doc,
                   st.session_state.session_id, use_streaming)


def _run_query(query, mode, depth, source_doc, session_id, use_streaming):
    scope_label = f"`{source_doc}`" if source_doc else "all documents"

    try:
        if mode in ("graph", "rag"):
            from backend.retrieval.graph_rag import GraphRAG
            rag = GraphRAG()
            if use_streaming:
                _run_streaming(rag, query, depth, source_doc, session_id)
            else:
                with st.spinner(f"Searching {scope_label}..."):
                    result = rag.retrieve(query, depth=depth,
                                          source_doc=source_doc, session_id=session_id)
                _show_graph_result(result, query)

        elif mode == "traditional":
            from backend.retrieval.traditional_rag import TraditionalRAG
            with st.spinner(f"Searching {scope_label}..."):
                result = TraditionalRAG().retrieve(query, source_doc=source_doc)
            _show_traditional_result(result)

        elif mode == "both":
            from backend.retrieval.graph_rag import GraphRAG
            from backend.retrieval.traditional_rag import TraditionalRAG
            with st.spinner(f"Searching {scope_label}..."):
                g = GraphRAG().retrieve(query, depth=depth,
                                        source_doc=source_doc, session_id=session_id)
                t = TraditionalRAG().retrieve(query, source_doc=source_doc)
            _show_comparison(query, g, t)

    except Exception as e:
        st.error(f"Query failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def _run_streaming(rag, query, depth, source_doc, session_id):
    """Real-time streaming via LangChain LCEL stream()."""
    st.markdown("### 🕸️ Graph RAG Answer  ⚡ streaming")
    placeholder = st.empty()
    full = ""
    try:
        for token in rag.stream_answer(query, depth=depth,
                                       source_doc=source_doc, session_id=session_id):
            full += token
            placeholder.markdown(full + "▌")
        placeholder.markdown(full)
    except Exception as e:
        placeholder.error(f"Streaming failed: {e}. Retrying without streaming...")
        result = rag.retrieve(query, depth=depth, source_doc=source_doc, session_id=session_id)
        _show_graph_result(result, query)


def _show_graph_result(result: dict, query: str):
    seeds     = result.get("seed_entities", [])
    answer    = result.get("answer", "")
    scope     = result.get("scope", "all documents")
    sentences = result.get("source_sentences", [])
    ctx       = result.get("graph_context", [])
    provider  = result.get("llm_provider", "unknown")

    st.markdown("### 🕸️ Graph RAG Answer")
    st.caption(
        f"Scope: **{scope}** | Entities: **{', '.join(seeds) if seeds else 'none'}** | "
        f"Provider: **{provider}**"
    )

    if answer:
        st.markdown(answer)
    else:
        st.warning("No answer generated. Try uploading relevant documents first.")

    if seeds:
        with st.expander("🌐 Knowledge graph used for this answer", expanded=True):
            try:
                from backend.database.neo4j_client import Neo4jClient
                from backend.visualization.graph_viz import GraphViz
                neo4j = Neo4jClient()
                all_nodes, all_edges, seen_n, seen_e = [], [], set(), set()
                for seed in seeds[:3]:
                    sg = neo4j.get_entity_subgraph(seed, depth=2)
                    for n in sg["nodes"]:
                        if n["name"] not in seen_n:
                            seen_n.add(n["name"]); all_nodes.append(n)
                    for e in sg["edges"]:
                        k = (e["source"], e["relationship"], e["target"])
                        if k not in seen_e:
                            seen_e.add(k); all_edges.append(e)
                if all_nodes:
                    fig = GraphViz().render_matplotlib(
                        all_nodes, all_edges,
                        title=f"Graph context for: {query[:50]}"
                    )
                    st.pyplot(fig)
                else:
                    st.info("No graph data to visualise for these entities.")
            except Exception as e:
                st.warning(f"Could not render graph: {e}")

    with st.expander("📋 Relationships traversed"):
        if ctx:
            for r in ctx[:15]:
                badge = "🔵" if r["depth"] == 1 else "🟡"
                st.markdown(
                    f"{badge} `{r['source']}` → **{r['relationship']}** → `{r['target']}`"
                )
        else:
            st.info("No relationships found for these entities.")

    with st.expander("📄 Source sentences from document"):
        if sentences:
            for s in sentences[:8]:
                st.markdown(f"> {s.get('sentence','')}")
                st.caption(
                    f"📁 {s.get('doc','?')} | "
                    f"{s['source']} → {s['relationship']} → {s['target']} | "
                    f"confidence: {s.get('confidence',0):.2f}"
                )
        else:
            st.info("No source sentences found.")


def _show_traditional_result(result: dict):
    method = result.get("method", "TF-IDF")
    st.markdown(f"### 📄 Traditional RAG Answer  `{method}`")
    st.markdown(result.get("answer", "No answer generated."))

    with st.expander("📋 Retrieved text chunks"):
        for c in result.get("retrieved_chunks", []):
            st.markdown(f"**Score: {c['score']:.3f}** | Source: {c.get('doc','?')}")
            st.markdown(f"> {c['text']}")
            st.divider()

    with st.expander("⚠️ Limitations of Traditional RAG"):
        for lim in result.get("limitations", []):
            st.markdown(f"- {lim}")


def _show_comparison(query: str, g: dict, t: dict):
    st.markdown("### ⚖️ Side-by-side Comparison")
    st.caption(f"Query: *{query}*")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.success("🕸️ Graph RAG")
        st.markdown(g.get("answer", "No answer."))
        st.caption(
            f"Entities: {', '.join(g.get('seed_entities',[]))} | "
            f"Hops: {sum(p['hops'] for p in g.get('paths_traversed',[]))}"
        )
    with c2:
        method = t.get("method", "TF-IDF")
        st.info(f"📄 Traditional RAG  `{method}`")
        st.markdown(t.get("answer", "No answer."))
        chunks = t.get("retrieved_chunks", [])
        best   = chunks[0]["score"] if chunks else 0
        st.caption(f"Chunks: {len(chunks)} | Best score: {best:.3f}")

    seeds = g.get("seed_entities", [])
    if seeds:
        with st.expander("🌐 Graph RAG — knowledge graph used"):
            try:
                from backend.database.neo4j_client import Neo4jClient
                from backend.visualization.graph_viz import GraphViz
                neo4j = Neo4jClient()
                all_nodes, all_edges, seen_n, seen_e = [], [], set(), set()
                for seed in seeds[:2]:
                    sg = neo4j.get_entity_subgraph(seed, depth=2)
                    for n in sg["nodes"]:
                        if n["name"] not in seen_n:
                            seen_n.add(n["name"]); all_nodes.append(n)
                    for e in sg["edges"]:
                        k = (e["source"], e["relationship"], e["target"])
                        if k not in seen_e:
                            seen_e.add(k); all_edges.append(e)
                if all_nodes:
                    fig = GraphViz().render_matplotlib(
                        all_nodes, all_edges,
                        title="Graph RAG — relationships traversed"
                    )
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Graph render failed: {e}")

    st.divider()
    st.subheader("📊 Scorecard")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Graph entities",       len(g.get("seed_entities", [])))
    s2.metric("Graph hops",           sum(p["hops"] for p in g.get("paths_traversed", [])))
    s3.metric("Traditional chunks",   len(t.get("retrieved_chunks", [])))
    chunks = t.get("retrieved_chunks", [])
    s4.metric("Best similarity",      f"{chunks[0]['score']:.3f}" if chunks else "0.000")
