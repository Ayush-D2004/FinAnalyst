"""
app.py – SEC Document Intelligence Streamlit UI
"""
import os
import tempfile
import time
import uuid
import warnings

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import transformers

HF_TOKEN = os.getenv("HF_TOKEN")
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", category=UserWarning, module="sec_parser.processing_steps.abstract_classes.abstract_processing_step")

transformers.utils.logging.set_verbosity_error()

from src.data.sqlite_store import SQLiteStore
from src.embeddings.encoder import DocumentEncoder
from src.embeddings.faiss_store import FAISSStore
from src.embeddings.reranker import DocumentReranker
from src.ingestion.edgar_client import EdgarClient
from src.ingestion.parser import EdgarParser
from src.preprocessing.chunker import SectionAwareChunker
from src.qa.citation import CitationManager
from src.qa.generator import QAGenerator

st.set_page_config(
    page_title="SEC Intelligence Assistant",
    page_icon="📈",
    layout="wide",
)

# ── System singleton (one load per session) ───────────────────────────────────

@st.cache_resource
def load_system() -> dict:
    return {
        "db":      SQLiteStore(),
        "chunker": SectionAwareChunker(),
        "encoder": DocumentEncoder(),
        "faiss":   FAISSStore(),
        "reranker":DocumentReranker(),
        "generator": QAGenerator(),
        "edgar_client": EdgarClient(),
        "parser":  EdgarParser(),
    }


sys = load_system()

# ── Shared ingest helper ──────────────────────────────────────────────────────

def ingest_file(file_path: str, doc_id: str, filename: str, company: str, year: str) -> int:
    """Parse → Chunk → Index into both SQLite and FAISS. Returns chunk count."""
    sections = sys["parser"].parse(file_path)
    chunks   = sys["chunker"].chunk_document(doc_id, sections)

    sys["db"].add_document(doc_id, filename, company, year)
    sys["db"].add_chunks(chunks)

    texts = [c["chunk_text"] for c in chunks]
    ids   = [c["chunk_id"]   for c in chunks]
    embs  = sys["encoder"].encode_documents(texts)
    sys["faiss"].add_embeddings(embs, ids)
    return len(chunks)


# ── Layout ─────────────────────────────────────────────────────────────────────

st.title("📈 SEC Document Intelligence")
st.caption("Powered by BAAI/bge-base · sec-parser · FAISS")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Management")

    docs = sys["db"].get_all_documents()
    doc_map = {f"{d['company_ticker']} — {d['filing_year']} ({d['filename']})": d["doc_id"] for d in docs}

    selected_label  = st.selectbox(
        "Active Document", options=["— select —"] + list(doc_map.keys())
    )
    active_doc_id = doc_map.get(selected_label)

    st.divider()

    # ── Option A: Fetch directly from SEC EDGAR ───────────────────────────────
    with st.expander("🌐 Fetch EDGAR 10-K", expanded=True):
        col1, col2 = st.columns(2)
        ticker_input = col1.text_input("Ticker", placeholder="AAPL")
        year_input   = col2.text_input("Year",   placeholder="2023")

        if st.button("📥 Fetch & Index"):
            if ticker_input and year_input:
                try:
                    year_int = int(year_input)
                except ValueError:
                    st.error("Year must be a 4-digit integer.")
                    st.stop()

                with st.spinner(f"Fetching {ticker_input.upper()} {year_int} 10-K from EDGAR …"):
                    t0 = time.time()
                    doc_id = f"{ticker_input.upper()}_{year_int}"
                    filing_path = sys["edgar_client"].get_10k_path(ticker_input, year_int)

                    if not filing_path:
                        st.error("Could not retrieve the filing. Check the ticker/year and try again.")
                    else:
                        n = ingest_file(
                            file_path=str(filing_path),
                            doc_id=doc_id,
                            filename=filing_path.name,
                            company=ticker_input.upper(),
                            year=str(year_int),
                        )
                        st.success(f"Indexed {n} chunks in {time.time()-t0:.1f}s!")
                        st.rerun()
            else:
                st.warning("Enter both a ticker and a year.")

    # ── Option B: Upload a PDF / HTML ─────────────────────────────────────────
    with st.expander("📄 Upload Your Own Document"):
        uploaded_file = st.file_uploader("PDF or HTML", type=["pdf", "html", "htm"])
        cmp = st.text_input("Company (optional)", value="UNKNOWN")
        yr  = st.text_input("Year (optional)", value="2024")

        if st.button("📤 Upload & Index"):
            if uploaded_file:
                suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                doc_id = str(uuid.uuid4())
                with st.spinner("Indexing …"):
                    t0 = time.time()
                    n = ingest_file(tmp_path, doc_id, uploaded_file.name, cmp, yr)
                st.success(f"Indexed {n} chunks in {time.time()-t0:.1f}s!")
                st.rerun()
            else:
                st.warning("Please select a file to upload.")


# ── Main Chat Area ─────────────────────────────────────────────────────────────
if not active_doc_id:
    st.info(
        "👈 Fetch an SEC 10-K by ticker/year **or** upload a document "
        "from the sidebar, then select it above to start chatting."
    )
    st.stop()

# Namespace message history per document
if "messages" not in st.session_state:
    st.session_state.messages = {}
if active_doc_id not in st.session_state.messages:
    st.session_state.messages[active_doc_id] = []

# Render existing messages
for msg in st.session_state.messages[active_doc_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📄 Source References"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['ref']} · {c['section']}**")
                    st.markdown(f"> {c['text'][:600]}…")

# Chat input
if prompt := st.chat_input("Ask a question about this filing …"):
    st.session_state.messages[active_doc_id].append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing filing …"):
            t0 = time.time()

            q_emb      = sys["encoder"].encode_queries(prompt)
            retrieved  = sys["faiss"].search(q_emb, top_k=25)
            chunk_ids  = [x[0] for x in retrieved]
            all_chunks = sys["db"].get_chunks_by_ids(chunk_ids)

            # Restrict to the active document
            doc_chunks = [c for c in all_chunks if c["doc_id"] == active_doc_id]

            if not doc_chunks:
                answer    = "I cannot answer this question — the document does not contain relevant information."
                citations = []
            else:
                texts    = [c["chunk_text"] for c in doc_chunks]
                reranked = sys["reranker"].rerank(prompt, texts, top_k=5)
                top_chunks = [doc_chunks[i] for i, _ in reranked]

                raw_answer = sys["generator"].generate_answer(prompt, top_chunks)
                payload    = CitationManager.attach_citations(raw_answer, top_chunks)
                answer     = payload["answer"]
                citations  = payload["citations"]

            elapsed = time.time() - t0

        st.markdown(answer)
        st.caption(f"⏱ {elapsed:.2f}s")

        if citations:
            with st.expander("📄 Source References"):
                for c in citations:
                    st.markdown(f"**{c['ref']} · {c['section']}**")
                    st.markdown(f"> {c['text'][:600]}…")

    st.session_state.messages[active_doc_id].append({
        "role":      "assistant",
        "content":   answer,
        "citations": citations,
    })
