# graph-rag

> A hybrid knowledge assistant combining Graph-based RAG with traditional vector search, built with FastAPI, Streamlit, and Neo4j.

---

##  Project Structure

```
graph-rag-assistant/
│
├── backend/                        # FastAPI Backend Server
│   ├── api/                        # API Routes
│   │   ├── routes/
│   │   │   ├── upload.py           # Document upload endpoints (US-02)
│   │   │   ├── graph.py            # Graph operations endpoints
│   │   │   ├── query.py            # Query endpoints
│   │   │   └── comparison.py       # RAG comparison endpoints
│   │   └── dependencies.py         # API dependencies
│   │
│   ├── core/                       # Core NLP Logic
│   │   ├── text_extractor.py       # Text extraction from documents (US-03) 
│   │   ├── entity_extractor.py     # Entity extraction (US-04) 
│   │   ├── relationship_extractor.py  # Relationship extraction (US-05) 
│   │   └── graph_builder.py        # Graph construction logic
│   │
│   ├── database/                   # Database Layer
│   │   ├── neo4j_client.py         # Neo4j connection & operations
│   │   ├── graph_queries.py        # Graph query functions
│   │   └── models.py               # Data models
│   │
│   ├── retrieval/                  # RAG Implementations
│   │   ├── traditional_rag.py      # Traditional RAG (vector search)
│   │   ├── graph_rag.py            # Graph-based RAG
│   │   └── query_processor.py      # Query processing logic
│   │
│   ├── visualization/              # Graph Visualization
│   │   ├── graph_viz.py            # Graph rendering
│   │   └── network_builder.py      # NetworkX graph builder
│   │
│   ├── utils/                      # Utility Functions
│   │   ├── file_handlers.py        # File handling utilities
│   │   └── helpers.py              # Helper functions
│   │
│   ├── config.py                   # Configuration settings
│   └── main.py                     # FastAPI entry point
│
├── frontend/                       # Streamlit Frontend
│   ├── pages/                      # Multi-page app
│   │   ├── upload_document.py      # Document upload page
│   │   ├── graph_explorer.py       # Graph visualization page
│   │   ├── query_assistant.py      # Query interface
│   │   └── rag_comparison.py       # RAG comparison page
│   │
│   ├── components/                 # Reusable components
│   │   ├── sidebar.py              # Navigation sidebar
│   │   ├── graph_viewer.py         # Graph viewer component
│   │   └── chat_interface.py       # Chat interface
│   │
│   ├── assets/
│   │   └── style.css               # Custom styles
│   │
│   └── streamlit_app.py            # Streamlit main app
│
├── data/                           # Data Storage
│   ├── uploads/                    # Uploaded documents
│   ├── processed/                  # Processed text files
│   └── sample_docs/                # Sample documents
│
├── scripts/                        # Utility Scripts
│   ├── init_neo4j.py               # Initialize Neo4j database
│   ├── load_sample_data.py         # Load sample data
│   └── test_script_one_five.py     # Test script (US-01 to US-05) 
│
├── tests/                          # Unit Tests
│   ├── test_extraction.py
│   ├── test_graph.py
│   └── test_queries.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
├── docker-compose.yml
├── .env
└── README.md
```

---

##  Completed Features

| User Story | Description | Status |
|------------|-------------|--------|
| US-01 | Project initialized with all dependencies |  
| US-02 | Document upload functionality |  
| US-03 | Text extraction (PDF, DOCX, TXT, MD) |  
| US-04 | Entity extraction |  |
| US-05 | Relationship extraction between entities |  

---

##  Local Setup

### Prerequisites

- Python **3.11** or **3.12**
- Git
- Docker *(optional, for Neo4j)*
- 8GB RAM minimum (16GB recommended)

---

### Step 1 — Clone the Repository

```bash
git clone git@github.com:arya23923/graph-rag.git
cd graph-rag-assistant
```

---

### Step 2 — Create a Virtual Environment

**Linux / Mac:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```env
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123

# OpenAI API (for advanced features)
OPENAI_API_KEY=your_key_here

# Application Settings
UPLOAD_DIR=./data/uploads
PROCESSED_DIR=./data/processed
MAX_FILE_SIZE=10485760
```

---

