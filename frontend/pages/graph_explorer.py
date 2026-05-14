"""
Graph Explorer — 100% dynamic from Neo4j.
Dropdown entities = whatever is actually stored in Neo4j from uploaded docs.
No hardcoded entity lists anywhere.
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _get_neo4j():
    from backend.database.neo4j_client import Neo4jClient
    return Neo4jClient()


def _load_entities_from_neo4j() -> list:
    """Fetch ALL entity names actually stored in Neo4j — from uploaded docs."""
    try:
        neo4j = _get_neo4j()
        entities = neo4j.get_all_entities(limit=500)
        names = sorted([e["name"] for e in entities if e.get("name")])
        return names
    except Exception as e:
        st.error(f"Could not load entities from Neo4j: {e}")
        return []


def render():
    st.title("🔭 Graph Explorer")
    st.markdown("Explore entity relationships from your **uploaded documents** — all data is live from Neo4j.")

    # ── Live entity list from Neo4j ───────────────────────────────────────
    with st.spinner("Loading entities from Neo4j..."):
        all_entities = _load_entities_from_neo4j()

    if not all_entities:
        st.warning("""
        ⚠️ No entities found in Neo4j yet.

        **Steps to fix:**
        1. Go to **📤 Upload & Build Graph** page
        2. Upload a document or paste text
        3. Click **Build Graph**
        4. Come back here — your entities will appear in the dropdown
        """)
        return

    st.success(f"✅ {len(all_entities)} entities loaded from Neo4j (from your uploaded documents)")

    tab1, tab2, tab3 = st.tabs(["🌐 Entity Subgraph", "🔀 Path Finder", "📊 Graph Stats"])

    # ── Tab 1: Entity subgraph ────────────────────────────────────────────
    with tab1:
        st.markdown("Select any entity from your uploaded documents to see its relationship graph.")

        col1, col2 = st.columns([3, 1])
        with col1:
            # Search box to filter entities
            search = st.text_input("🔍 Search entity:", placeholder="Type to filter...")
            filtered = [e for e in all_entities if search.lower() in e.lower()] if search else all_entities

            if not filtered:
                st.warning(f"No entity found matching '{search}'")
                return

            entity = st.selectbox(
                f"Select entity ({len(filtered)} found):",
                filtered,
                help="These are real entities extracted from your uploaded documents"
            )
        with col2:
            depth = st.slider("Traversal depth:", 1, 4, 2)

        if st.button("🔍 Explore Entity", use_container_width=True, type="primary"):
            with st.spinner(f"Fetching graph for '{entity}' from Neo4j..."):
                _show_entity_graph(entity, depth)

    # ── Tab 2: Path finder ────────────────────────────────────────────────
    with tab2:
        st.markdown("Find the shortest path between any two entities in your uploaded documents.")

        c1, c2 = st.columns(2)
        with c1:
            search_src = st.text_input("Search source:", placeholder="Type to filter...", key="src_search")
            src_filtered = [e for e in all_entities if search_src.lower() in e.lower()] if search_src else all_entities
            src = st.selectbox("Source entity:", src_filtered, key="src_sel")
        with c2:
            search_tgt = st.text_input("Search target:", placeholder="Type to filter...", key="tgt_search")
            tgt_filtered = [e for e in all_entities if search_tgt.lower() in e.lower()] if search_tgt else all_entities
            tgt = st.selectbox("Target entity:", tgt_filtered, key="tgt_sel")

        max_d = st.slider("Max path depth:", 2, 6, 4, key="path_depth")

        if st.button("🗺️ Find Path", use_container_width=True):
            if src == tgt:
                st.warning("Source and target must be different entities.")
            else:
                _show_path(src, tgt, max_d)

    # ── Tab 3: Stats ──────────────────────────────────────────────────────
    with tab3:
        _show_stats(all_entities)


# ── helpers ────────────────────────────────────────────────────────────────

def _show_entity_graph(entity: str, depth: int):
    from backend.visualization.graph_viz import GraphViz

    neo4j = _get_neo4j()

    # Get real subgraph from Neo4j
    subgraph = neo4j.get_entity_subgraph(entity, depth)
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    if not nodes:
        st.warning(f"No connections found for '{entity}' in Neo4j. Try a different entity or lower the depth.")
        return

    st.markdown(f"**{len(nodes)} nodes · {len(edges)} edges** from Neo4j (depth={depth})")

    # Draw graph
    viz = GraphViz()
    fig = viz.render_matplotlib(
        nodes, edges,
        title=f"Entity Graph: {entity} (depth {depth})"
    )
    st.pyplot(fig)

    # Show raw relationships table
    with st.expander("📋 All relationships for this entity"):
        related = neo4j.get_related_entities(entity, depth)
        if related:
            for r in related:
                badge = "🔵" if r["depth"] == 1 else "🟡"
                st.markdown(
                    f"{badge} `{r['source']}` → **{r['relationship']}** → `{r['target']}` "
                    f"*(hop {r['depth']})*"
                )
        else:
            st.info("No relationships found.")

    # Show connected entities list
    with st.expander("📎 Connected entities"):
        connected = [n["name"] for n in nodes if n["name"] != entity]
        if connected:
            cols = st.columns(3)
            for i, name in enumerate(connected):
                cols[i % 3].markdown(f"- {name}")
        else:
            st.info("No connected entities found at this depth.")


def _show_path(src: str, tgt: str, max_depth: int):
    from backend.visualization.graph_viz import GraphViz

    neo4j = _get_neo4j()
    paths = neo4j.find_path(src, tgt, max_depth)

    if not paths:
        st.warning(f"No path found between **{src}** and **{tgt}** within {max_depth} hops.")
        st.info("Try increasing the max path depth, or check if both entities are connected in your documents.")
        return

    st.success(f"Found {len(paths)} path(s) between **{src}** and **{tgt}**")

    for i, path in enumerate(paths, 1):
        st.markdown(f"**Path {i}** — {path['length']} hops")
        path_str = path["nodes"][0]
        for rel, node in zip(path["relationships"], path["nodes"][1:]):
            path_str += f" → [{rel}] → {node}"
        st.code(path_str)

    # Draw path graph
    viz = GraphViz()
    all_node_names = set()
    all_edges = []
    for path in paths:
        for n in path["nodes"]:
            all_node_names.add(n)
        for s, r, t in zip(path["nodes"], path["relationships"], path["nodes"][1:]):
            all_edges.append({"source": s, "target": t, "relationship": r})

    node_list = [{"name": n, "type": "ENTITY", "confidence": 0.9} for n in all_node_names]
    fig = viz.render_matplotlib(node_list, all_edges, title=f"Path: {src} → {tgt}")
    st.pyplot(fig)


def _show_stats(all_entities: list):
    neo4j = _get_neo4j()
    stats = neo4j.get_graph_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Entities",       stats.get("entities", len(all_entities)))
    c2.metric("Total Relationships",  stats.get("relationships", 0))
    c3.metric("Neo4j Status",         "✅ Connected" if stats.get("connected") else "⚠️ Mock mode")

    # Show all entities from Neo4j grouped by type
    neo4j_entities = neo4j.get_all_entities(limit=500)
    if neo4j_entities:
        # Group by type
        by_type = {}
        for e in neo4j_entities:
            t = e.get("type", "UNKNOWN")
            by_type.setdefault(t, []).append(e["name"])

        st.subheader("📋 All entities in Neo4j (from your uploaded docs)")
        for etype, names in sorted(by_type.items()):
            with st.expander(f"{etype} — {len(names)} entities"):
                cols = st.columns(3)
                for i, name in enumerate(sorted(names)):
                    cols[i % 3].markdown(f"- {name}")

    # Show source documents
    st.subheader("📄 Source documents indexed")
    try:
        with neo4j._driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.source IS NOT NULL "
                "RETURN DISTINCT n.source AS src, count(n) AS cnt "
                "ORDER BY cnt DESC"
            )
            docs = [(r["src"], r["cnt"]) for r in result]
            if docs:
                for doc, cnt in docs:
                    st.markdown(f"- **{doc}** — {cnt} entities")
            else:
                st.info("No source documents found.")
    except Exception:
        st.info("Connect Neo4j to see source documents.")
