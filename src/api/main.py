import os
import uuid
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.data.sqlite_store import SQLiteStore
from src.ingestion.parser import DocumentParser
from src.preprocessing.chunker import SectionAwareChunker
from src.embeddings.encoder import DocumentEncoder
from src.embeddings.faiss_store import FAISSStore
from src.embeddings.reranker import DocumentReranker
from src.qa.generator import QAGenerator
from src.qa.citation import CitationManager
from src import config

app = FastAPI(title="SEC RAG Assistant API")

# Global instances initialized on startup
# Doing lazy initialization to allow safe import without loading weights
db_store = None
chunker = None
encoder = None
faiss_store = None
reranker = None
generator = None

@app.on_event("startup")
async def startup_event():
    global db_store, chunker, encoder, faiss_store, reranker, generator
    db_store = SQLiteStore()
    chunker = SectionAwareChunker()
    encoder = DocumentEncoder()
    faiss_store = FAISSStore()
    reranker = DocumentReranker()
    generator = QAGenerator()

class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    raw_context: List[dict]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), company: str = Form("Unknown"), year: str = Form("Unknown")):
    # Save file temporarily
    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    os.makedirs("/tmp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    doc_id = str(uuid.uuid4())
    
    # 1. Parse
    sections = DocumentParser.parse_file(temp_path)
    
    # 2. Chunk
    chunks = chunker.chunk_document(doc_id, sections)
    
    # 3. Store in DB
    db_store.add_document(doc_id, file.filename, company, year)
    db_store.add_chunks(chunks)
    
    # 4. Embed & Vector Store
    chunk_texts = [c["chunk_text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    
    embeddings = encoder.encode_documents(chunk_texts)
    faiss_store.add_embeddings(embeddings, chunk_ids)
    
    return {"status": "success", "doc_id": doc_id, "chunks_indexed": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query_system(req: QueryRequest):
    # 1. Encode query
    q_emb = encoder.encode_queries(req.query)
    
    # 2. Initial retrieval from FAISS
    # We retrieve 5x more for reranker precision
    retrieved = faiss_store.search(q_emb, top_k=req.top_k * 5)
    
    if not retrieved:
        return QueryResponse(answer="No documents found in index.", citations=[], raw_context=[])
        
    chunk_ids = [x[0] for x in retrieved]
    
    # Fetch from SQLite
    chunks_meta = db_store.get_chunks_by_ids(chunk_ids)
    
    # Optional filtering: strictly by doc_id if provided
    if req.doc_id:
        chunks_meta = [c for c in chunks_meta if c["doc_id"] == req.doc_id]
        
    if not chunks_meta:
        return QueryResponse(answer="Document contains no relevant context.", citations=[], raw_context=[])
        
    # 3. Rerank
    texts_to_rerank = [c["chunk_text"] for c in chunks_meta]
    reranked = reranker.rerank(req.query, texts_to_rerank, top_k=req.top_k)
    
    top_chunks = [chunks_meta[idx] for idx, score in reranked]
    
    # 4. Generate answer
    raw_answer = generator.generate_answer(req.query, top_chunks)
    
    # 5. Build citations
    final_payload = CitationManager.attach_citations(raw_answer, top_chunks)
    
    return QueryResponse(**final_payload)

@app.get("/documents")
async def list_documents():
    return db_store.get_all_documents()
