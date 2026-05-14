"""
Tech Domain Graph RAG — Streamlit entry point.
Run: streamlit run frontend/streamlit_app.py
"""
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Tech Graph RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  .stMetric { background: #1E1E2E; border-radius: 8px; padding: 8px; }
  div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

from frontend.components.sidebar import render_sidebar
from frontend.pages import upload_document, graph_explorer, query_assistant, rag_comparison

page = render_sidebar()

if page == "🏠 Home":
    st.title("🧠 Tech Graph RAG Knowledge Assistant")
    st.markdown("""
    A **Graph-based Retrieval Augmented Generation** system.  
    Upload any tech document → entities and relationships are extracted → stored in Neo4j → queryable as a graph.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.info("**📤 Upload & Build Graph**\nUpload any doc — entities and relationships extracted into Neo4j")
    c2.info("**🔭 Graph Explorer**\nExplore entities from YOUR uploaded docs — 100% live from Neo4j")
    c3.info("**💬 Query Assistant**\nAsk relationship-based questions about your documents")
    c4.info("**⚖️ RAG Comparison**\nSide-by-side: Graph RAG vs Traditional RAG")

    st.divider()

    # Show live graph from Neo4j — only what's actually stored
    from backend.database.neo4j_client import Neo4jClient
    from backend.visualization.graph_viz import GraphViz

    neo4j = Neo4jClient()
    stats = neo4j.get_graph_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Entities in Neo4j",       stats.get("entities", 0))
    col2.metric("Relationships in Neo4j",  stats.get("relationships", 0))
    col3.metric("Neo4j Status", "✅ Connected" if stats.get("connected") else "⚠️ Mock mode")

    if stats.get("entities", 0) > 0:
        st.subheader("📊 Your Knowledge Graph (live from Neo4j)")

        # Get ALL real entities from Neo4j
        all_entities = neo4j.get_all_entities(limit=500)

        # Build full graph from all stored entities
        all_nodes = []
        all_edges = []
        seen_nodes = set()
        seen_edges = set()

        # Sample up to 10 entities for home view
        sample_entities = [e["name"] for e in all_entities[:10]]

        for entity_name in sample_entities:
            sg = neo4j.get_entity_subgraph(entity_name, depth=1)
            for n in sg["nodes"]:
                if n["name"] not in seen_nodes:
                    seen_nodes.add(n["name"])
                    all_nodes.append(n)
            for e in sg["edges"]:
                key = (e["source"], e["relationship"], e["target"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    all_edges.append(e)

        if all_nodes:
            viz = GraphViz()
            fig = viz.render_matplotlib(
                all_nodes, all_edges,
                title=f"Knowledge Graph — {len(all_nodes)} entities from your documents",
                figsize=(16, 9)
            )
            st.pyplot(fig)
        else:
            st.info("Graph is empty. Upload a document to see it here.")
    else:
        st.info("""
        👆 **No documents uploaded yet.**

        Get started:
        1. Go to **📤 Upload & Build Graph** in the sidebar
        2. Upload a document or paste text
        3. Click **Build Graph**
        4. Come back here to see your knowledge graph!
        """)

elif page == "📤 Upload & Build Graph":
    upload_document.render()

elif page == "🔭 Graph Explorer":
    graph_explorer.render()

elif page == "💬 Query Assistant":
    query_assistant.render()

elif page == "⚖️ RAG Comparison":
    rag_comparison.render()
