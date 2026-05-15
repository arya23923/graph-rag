# LangChain Integration Guide

## What changed and why

This document maps every original file to its LangChain-powered replacement.

---

## File-by-file changes

### `backend/retrieval/llm_client.py` ← **rewritten**

| Before | After |
|--------|-------|
| Raw `requests.post` to Groq/OpenAI | `langchain_groq.ChatGroq` + `langchain_openai.ChatOpenAI` |
| Manual JSON parsing of API response | `StrOutputParser` |
| No streaming | `stream_llm_answer()` generator via LCEL `.stream()` |
| New LLM = new function | New LLM = add one line to `_invoke` loop |

**Key win:** LangChain handles retries, timeouts, and provider-agnostic interfaces automatically.

---

### `backend/retrieval/graph_rag.py` ← **rewritten**

| Before | After |
|--------|-------|
| `_generate_answer()` calls raw LLM client | LangChain LCEL chain: `_PROMPT \| llm \| StrOutputParser()` |
| No conversation history | `ConversationBufferMemory` — remembers previous questions |
| No streaming | `stream_answer()` method yields tokens in real-time |
| Prompt built as plain strings | `ChatPromptTemplate` with typed `{question}`, `{context}`, `{chat_history}` variables |

**Key win:** Multi-turn conversation context improves answer quality for follow-up questions.

---

### `backend/retrieval/traditional_rag.py` ← **upgraded**

| Before | After |
|--------|-------|
| TF-IDF only (keyword matching) | `HuggingFaceEmbeddings` (sentence-transformers, free local) |
| `sklearn` cosine similarity | LangChain FAISS vector store with `similarity_search_with_score` |
| Misses semantic meaning | Understands synonyms and paraphrases |
| TF-IDF still available | Automatic fallback when embeddings unavailable |

**Key win:** "machine learning" now matches "neural network training" — not just exact keywords.

---

### `backend/core/entity_extractor.py` ← **upgraded**

| Before | After |
|--------|-------|
| spaCy NER + regex only | + LangChain `PromptTemplate \| ChatGroq \| StrOutputParser` chain |
| Domain entities only via TECH_BOOST_TERMS | LLM extracts entity type from any domain via JSON output |
| All-or-nothing extraction | LLM runs on first 3000 chars, spaCy/regex covers the rest |

**Key win:** Better entity recognition for non-tech documents (legal, medical, scientific).

---

### `backend/core/document_processor.py` ← **new file**

New LangChain document loading pipeline:

- `PyPDFLoader` for PDFs (preserves page structure)  
- `Docx2txtLoader` for Word documents  
- `TextLoader` with auto encoding detection  
- `UnstructuredFileLoader` as generic fallback  
- `RecursiveCharacterTextSplitter` (chunk_size=1200, overlap=150) for smart chunking  
- `GraphBuilder` now processes chunks in batch → better entity recall on large docs  

---

### `backend/core/graph_builder.py` ← **upgraded**

| Before | After |
|--------|-------|
| Single-pass over full text | Chunk-by-chunk entity extraction (higher recall) |
| `TextExtractor` hardcoded | `DocumentProcessor` (LangChain) with `TextExtractor` fallback |
| Entity dedup not guaranteed | Dict-keyed dedup keeps highest-confidence entity per name |

---

### `frontend/pages/query_assistant.py` ← **upgraded**

| Before | After |
|--------|-------|
| Wait for full answer | ⚡ Real-time token streaming (toggle in UI) |
| Stateless | 🧠 Conversation memory displayed + clearable per session |
| No LLM provider info | Shows which provider answered (Groq / OpenAI / rule-based) |
| Traditional RAG shows TF-IDF label | Shows actual method used (FAISS semantic / TF-IDF) |

---

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Install spaCy model (optional but recommended)
python -m spacy download en_core_web_sm

# Copy and fill in your keys
cp .env.example .env
# Edit .env — add GROQ_API_KEY (free at console.groq.com)

# Run
streamlit run frontend/streamlit_app.py
```

## LangChain packages added

| Package | Purpose |
|---------|---------|
| `langchain>=0.2.0` | Core LCEL, memory, text splitters |
| `langchain-core>=0.2.0` | Prompts, output parsers, runnables |
| `langchain-community>=0.2.0` | Document loaders, FAISS, HuggingFace embeddings |
| `langchain-groq>=0.1.3` | `ChatGroq` — free LLaMA 3 |
| `langchain-openai>=0.1.0` | `ChatOpenAI` — GPT-3.5/4 fallback |
| `langchain-huggingface>=0.0.3` | Local `HuggingFaceEmbeddings` |
| `faiss-cpu>=1.7.4` | Vector store for semantic search |
| `sentence-transformers>=2.2.0` | Free local embedding model |

## Optional: LangSmith tracing

To trace and debug LangChain chains:
1. Create account at https://smith.langchain.com
2. Set in `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key
   ```
All chain calls will appear in the LangSmith dashboard.
