"""Upload page -- Graph RAG. Supports ANY file type (txt, pdf, docx, md, csv, json, xlsx)."""

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
    "Business / Legal": (
        "Apple Inc was founded by Steve Jobs and Steve Wozniak in Cupertino, California. "
        "Apple acquired Beats Electronics and partners with TSMC for chip manufacturing. "
        "Microsoft competes with Apple in the consumer electronics market. "
        "The EU regulates Apple under the Digital Markets Act, which is based in Brussels."
    ),
    "Medical / Research": (
        "Aspirin is used to treat inflammation and reduces fever. "
        "Pfizer developed the BNT162b2 vaccine in partnership with BioNTech. "
        "The FDA regulates pharmaceutical drugs and approved Paxlovid for COVID-19 treatment. "
        "Moderna uses mRNA technology to develop vaccines that target viral proteins."
    ),
}

# Supported file types for upload
SUPPORTED_TYPES = ["txt", "pdf", "md", "docx", "csv", "json", "xlsx", "xls"]


def render():
    st.title("Upload & Build Knowledge Graph")

    st.markdown(
        "Upload **any** document. The pipeline extracts entities "
        "(people, orgs, places, products, concepts) and relationships, "
        "then stores them in Neo4j for graph-based retrieval."
    )

    tab1, tab2 = st.tabs(["Upload File", "Paste / Sample Text"])

    # ---- Tab 1: File upload -----------------------------------------------
    with tab1:
        st.info("Supported: .txt  .pdf  .md  .docx  .csv  .json  .xlsx")

        uploaded = st.file_uploader(
            "Choose a file",
            type=SUPPORTED_TYPES,
            help="Any document -- tech docs, business reports, research papers, legal texts, etc."
        )

        if uploaded:
            st.success(f"Loaded: {uploaded.name} ({uploaded.size:,} bytes)")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Process & Build Graph",
                    key="upload_btn",
                    use_container_width=True
                ):
                    _process_uploaded_file(uploaded)

            with col2:
                if st.button(
                    "Preview Extracted Text",
                    key="preview_btn",
                    use_container_width=True
                ):
                    _preview_file(uploaded)

    # ---- Tab 2: Text input ------------------------------------------------
    with tab2:
        sample_choice = st.selectbox(
            "Load a sample:",
            ["-- custom --"] + list(SAMPLE_TEXTS.keys())
        )

        default_text = (
            SAMPLE_TEXTS.get(sample_choice, "")
            if sample_choice != "-- custom --"
            else ""
        )

        text = st.text_area(
            "Paste your document text here:",
            value=default_text,
            height=220
        )

        doc_name = st.text_input(
            "Document name:",
            value="pasted_document"
        )

        if st.button("Build Graph from Text", key="text_btn") and text.strip():
            with st.spinner("Extracting entities and relationships..."):
                try:
                    r = requests.post(
                        f"{API}/api/graph/build",
                        json={
                            "text": text,
                            "source_doc": doc_name
                        },
                        timeout=60
                    )

                    if r.status_code == 200:
                        _show_build_result(r.json())
                    else:
                        st.error(f"API error: {r.text}")

                except requests.exceptions.ConnectionError:
                    _local_build(text, doc_name)


# --------------------------------------------------------------------------
# Processing functions
# --------------------------------------------------------------------------

def _process_uploaded_file(uploaded):
    """
    Reads actual file content and sends to backend.
    Falls back to local processing if backend fails.
    """

    with st.spinner(f"Processing {uploaded.name}..."):

        try:
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream"
                )
            }

            r_up = requests.post(
                f"{API}/api/upload/",
                files=files,
                timeout=30
            )

            if r_up.status_code == 200:

                saved_filename = r_up.json()["file_info"]["saved_filename"]

                st.info(f"Saved as {saved_filename}")

                r_build = requests.post(
                    f"{API}/api/graph/build-file",
                    json={"filename": saved_filename},
                    timeout=120
                )

                if r_build.status_code == 200:
                    _show_build_result(r_build.json())
                else:
                    st.warning(
                        f"Build via API failed ({r_build.text}). "
                        "Running locally..."
                    )
                    _local_build_from_bytes(uploaded)

            else:
                st.warning(
                    f"Upload failed ({r_up.text}). Running locally..."
                )
                _local_build_from_bytes(uploaded)

        except requests.exceptions.ConnectionError:
            st.warning("Backend not running -- processing locally.")
            _local_build_from_bytes(uploaded)


def _preview_file(uploaded):
    """Preview extracted text."""

    with st.spinner("Extracting preview..."):

        try:
            text = _extract_text_from_bytes(
                uploaded.name,
                uploaded.getvalue()
            )

            st.subheader("Extracted text (first 500 chars)")

            st.code(
                text[:500] + ("..." if len(text) > 500 else "")
            )

        except Exception as e:
            st.error(f"Could not preview: {e}")


def _local_build_from_bytes(uploaded):
    """
    Extract text locally and build graph.
    """

    try:
        text = _extract_text_from_bytes(
            uploaded.name,
            uploaded.getvalue()
        )

        if not text.strip():
            st.error("Could not extract any text from the file.")
            return

        st.info(
            f"Extracted {len(text):,} characters from {uploaded.name}"
        )

        _local_build(text, uploaded.name)

    except Exception as e:
        st.error(f"Local extraction failed: {e}")


def _extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """
    Extract text from supported file types.
    """

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

                return "\n".join(
                    p.extract_text() or ""
                    for p in reader.pages
                )

            except Exception as e:
                raise RuntimeError(f"PDF extraction failed: {e}")

    elif ext == ".docx":

        from docx import Document

        doc = Document(io.BytesIO(raw_bytes))

        return "\n".join(
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        )

    elif ext == ".csv":

        import csv
        import io as _io

        text_io = _io.StringIO(
            raw_bytes.decode("utf-8", errors="replace")
        )

        rows = list(csv.reader(text_io))

        return "\n".join(
            ", ".join(row)
            for row in rows
        )

    elif ext == ".json":

        import json

        data = json.loads(
            raw_bytes.decode("utf-8", errors="replace")
        )

        return _flatten_json(data)

    elif ext in (".xlsx", ".xls"):

        try:
            import openpyxl

            wb = openpyxl.load_workbook(
                io.BytesIO(raw_bytes),
                read_only=True
            )

            lines = []

            for ws in wb.worksheets:

                for row in ws.iter_rows(values_only=True):

                    line = ", ".join(
                        str(c)
                        for c in row
                        if c is not None
                    )

                    if line:
                        lines.append(line)

            return "\n".join(lines)

        except Exception as e:
            raise RuntimeError(f"Excel extraction failed: {e}")

    else:
        # Fallback
        return raw_bytes.decode("utf-8", errors="replace")


def _flatten_json(obj, depth=0, max_depth=4) -> str:
    """
    Flatten JSON into readable text.
    """

    if depth > max_depth:
        return str(obj)

    if isinstance(obj, dict):

        parts = []

        for k, v in obj.items():
            parts.append(
                f"{k}: {_flatten_json(v, depth + 1, max_depth)}"
            )

        return ". ".join(parts)

    elif isinstance(obj, list):

        return ". ".join(
            _flatten_json(item, depth + 1, max_depth)
            for item in obj[:50]
        )

    else:
        return str(obj)


def _local_build(text: str, doc_name: str):
    """
    Run extraction pipeline locally.
    """

    try:
        sys.path.insert(0, ".")

        from backend.core.entity_extractor import EntityExtractor
        from backend.core.relationship_extractor import RelationshipExtractor

        ent_ex = EntityExtractor(domain=None)
        rel_ex = RelationshipExtractor()

        entities = ent_ex.extract(
            text,
            source_doc=doc_name
        )

        rels = rel_ex.extract(
            text,
            entities,
            source_doc=doc_name
        )

        _show_build_result({
            "success": True,
            "entities_count": len(entities),
            "relationships_count": len(rels),
            "entities": entities,
            "relationships": rels,
            "graph_result": {
                "status": "local mode (Neo4j not required)"
            }
        })

    except Exception as e:
        st.error(f"Local build failed: {e}")


def _show_build_result(result: dict):

    if not result.get("success", True):
        st.error(result.get("error", "Unknown error"))
        return

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Entities Found",
        result.get("entities_count", 0)
    )

    col2.metric(
        "Relationships Found",
        result.get("relationships_count", 0)
    )

    col3.metric(
        "Graph Status",
        result.get("graph_result", {}).get("status", "ok")
    )

    with st.expander("Entities extracted (up to 50)"):

        for e in result.get("entities", [])[:50]:

            st.markdown(
                f"- **{e['name']}** "
                f"-- confidence: {e.get('confidence', 0):.2f}"
            )

    with st.expander("Relationships extracted (up to 50)"):

        for r in result.get("relationships", [])[:50]:

            st.markdown(
                f"- "
                f"{r.get('source', '?')} "
                f"-> **{r['relationship']}** -> "
                f"{r.get('target', '?')} "
                f"({r.get('confidence', 0):.2f})"
            )