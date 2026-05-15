"""Upload & Build Knowledge Graph page — Tech Domain Graph RAG."""
import streamlit as st
import requests
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

API = "http://localhost:8000"

SAMPLE_TEXTS = {
    "Cloud & DevOps": (
        "Kubernetes orchestrates Docker containers and runs on AWS, Azure, and GCP. "
        "Terraform integrates with Kubernetes to provision infrastructure declaratively. "
        "Jenkins deploys applications using Docker and works with Ansible for configuration. "
        "AWS supports Kubernetes via EKS and integrates with Terraform for provisioning."
    ),
    "AI & Data Stack": (
        "LangChain integrates with OpenAI and uses Neo4j as a graph store for retrieval. "
        "Python supports LangChain, FastAPI, and NetworkX for graph analysis. "
        "Kafka integrates with Spark for real-time data processing and connects to Elasticsearch. "
        "Hugging Face works with LangChain to serve open-source language models."
    ),
    "Backend & APIs": (
        "FastAPI is built with Python and connects to Neo4j for graph-based queries. "
        "Redis supports FastAPI for caching and connects to MongoDB for persistent storage. "
        "PostgreSQL integrates with FastAPI via SQLAlchemy and runs on AWS RDS. "
        "Elasticsearch connects to Kafka and supports full-text search for backend services."
    ),
}

SUPPORTED_TYPES = ["txt", "pdf", "md", "docx", "csv", "json"]


def render():
    st.title("📤 Upload & Build Knowledge Graph")
    st.markdown(
        "Upload a tech document. The pipeline extracts entities and relationships, "
        "then stores them in Neo4j for graph-based retrieval."
    )

    tab1, tab2 = st.tabs(["📄 Upload File", "✍️ Paste / Sample Text"])

    with tab1:
        st.info("Supported formats: .txt  .pdf  .md  .docx  .csv  .json")
        uploaded = st.file_uploader(
            "Choose a file",
            type=SUPPORTED_TYPES,
            help="Any tech document — architecture docs, README files, API docs, etc."
        )
        if uploaded:
            st.success(f"Loaded: {uploaded.name} ({uploaded.size:,} bytes)")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Process & Build Graph", key="upload_btn", use_container_width=True):
                    _process_uploaded_file(uploaded)
            with col2:
                if st.button("👁️ Preview Extracted Text", key="preview_btn", use_container_width=True):
                    _preview_file(uploaded)

    with tab2:
        sample_choice = st.selectbox("Load a sample:", ["— custom —"] + list(SAMPLE_TEXTS.keys()))
        default_text = SAMPLE_TEXTS.get(sample_choice, "") if sample_choice != "— custom —" else ""
        text = st.text_area("Paste your document text here:", value=default_text, height=220)
        doc_name = st.text_input("Document name:", value="pasted_document")

        if st.button("🚀 Build Graph from Text", key="text_btn") and text.strip():
            with st.spinner("Extracting entities and relationships..."):
                try:
                    r = requests.post(
                        f"{API}/api/graph/build",
                        json={"text": text, "source_doc": doc_name},
                        timeout=60
                    )
                    if r.status_code == 200:
                        _show_build_result(r.json())
                    else:
                        st.error(f"API error: {r.text}")
                except requests.exceptions.ConnectionError:
                    _local_build(text, doc_name)


# ── helpers ────────────────────────────────────────────────────────────────

def _process_uploaded_file(uploaded):
    with st.spinner(f"Processing {uploaded.name}..."):
        try:
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")}
            r_up = requests.post(f"{API}/api/upload/", files=files, timeout=30)
            if r_up.status_code == 200:
                saved_filename = r_up.json()["file_info"]["saved_filename"]
                st.info(f"Saved as: {saved_filename}")
                r_build = requests.post(
                    f"{API}/api/graph/build-file",
                    json={"filename": saved_filename},
                    timeout=120
                )
                if r_build.status_code == 200:
                    _show_build_result(r_build.json())
                else:
                    st.warning("Build via API failed. Running locally...")
                    _local_build_from_bytes(uploaded)
            else:
                st.warning(f"Upload failed. Running locally...")
                _local_build_from_bytes(uploaded)
        except requests.exceptions.ConnectionError:
            st.warning("Backend not running — processing locally.")
            _local_build_from_bytes(uploaded)


def _preview_file(uploaded):
    with st.spinner("Extracting preview..."):
        try:
            text = _extract_text_from_bytes(uploaded.name, uploaded.getvalue())
            st.subheader("Extracted text (first 500 chars)")
            st.code(text[:500] + ("..." if len(text) > 500 else ""))
        except Exception as e:
            st.error(f"Could not preview: {e}")


def _local_build_from_bytes(uploaded):
    try:
        text = _extract_text_from_bytes(uploaded.name, uploaded.getvalue())
        if not text.strip():
            st.error("Could not extract any text from the file.")
            return
        st.info(f"Extracted {len(text):,} characters from {uploaded.name}")
        _local_build(text, uploaded.name)
    except Exception as e:
        st.error(f"Local extraction failed: {e}")


def _extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext in (".txt", ".md"):
        return raw_bytes.decode("utf-8", errors="replace")

    elif ext == ".pdf":
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        except ImportError:
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw_bytes))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception as e:
                raise RuntimeError(f"PDF extraction failed: {e}")

    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".csv":
        import csv
        text_io = io.StringIO(raw_bytes.decode("utf-8", errors="replace"))
        rows = list(csv.reader(text_io))
        return "\n".join(", ".join(row) for row in rows)

    elif ext == ".json":
        import json
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
        return _flatten_json(data)

    else:
        return raw_bytes.decode("utf-8", errors="replace")


def _flatten_json(obj, depth=0, max_depth=4) -> str:
    if depth > max_depth:
        return str(obj)
    if isinstance(obj, dict):
        return ". ".join(f"{k}: {_flatten_json(v, depth+1, max_depth)}" for k, v in obj.items())
    elif isinstance(obj, list):
        return ". ".join(_flatten_json(item, depth+1, max_depth) for item in obj[:50])
    else:
        return str(obj)


def _local_build(text: str, doc_name: str):
    try:
        from backend.core.entity_extractor import EntityExtractor
        from backend.core.relationship_extractor import RelationshipExtractor

        ent_ex = EntityExtractor(domain="tech")
        rel_ex = RelationshipExtractor()
        entities = ent_ex.extract(text, source_doc=doc_name)
        rels = rel_ex.extract(text, entities, source_doc=doc_name)
        _show_build_result({
            "success": True,
            "entities_count": len(entities),
            "relationships_count": len(rels),
            "entities": entities,
            "relationships": rels,
            "graph_result": {"status": "local mode (Neo4j not required)"}
        })
    except Exception as e:
        st.error(f"Local build failed: {e}")


def _show_build_result(result: dict):
    if not result.get("success", True):
        st.error(result.get("error", "Unknown error"))
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Entities Found",       result.get("entities_count", 0))
    col2.metric("Relationships Found",  result.get("relationships_count", 0))
    col3.metric("Graph Status",         result.get("graph_result", {}).get("status", "ok"))

    with st.expander("🔍 Entities extracted"):
        for e in result.get("entities", [])[:50]:
            st.markdown(f"- **{e['name']}** `{e.get('type','?')}` — confidence: {e.get('confidence', 0):.2f}")

    with st.expander("🔗 Relationships extracted"):
        for r in result.get("relationships", [])[:50]:
            st.markdown(f"- `{r['source']}` → **{r['relationship']}** → `{r['target']}`")
