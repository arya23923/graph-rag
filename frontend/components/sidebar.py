import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Tech Graph RAG")
        st.caption("Knowledge Graph Assistant — Tech Domain")
        st.divider()

        page = st.radio(
            "Navigate",
            ["🏠 Home", "📤 Upload & Build Graph", "🔭 Graph Explorer",
             "💬 Query Assistant", "⚖️ RAG Comparison"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Legend**")
        st.markdown("🔵 Technology  🟠 Organisation  🟢 Person")
        st.markdown("**Relationship types:**")
        st.caption("INTEGRATES_WITH · RUNS_ON · ORCHESTRATES · BUILT_WITH · USES · CONNECTS_TO")
        st.divider()
        st.markdown("[📚 Docs](https://neo4j.com/docs/)  |  [🐙 GitHub](#)")

    return page
