import sqlite3
import uuid
from typing import List, Dict, Optional
from src import config

class SQLiteStore:
    def __init__(self, db_path: str = str(config.DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT,
                    company_ticker TEXT,
                    filing_year TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    section_name TEXT,
                    chunk_text TEXT,
                    token_count INTEGER,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                )
            ''')
            conn.commit()

    def add_document(self, doc_id: str, filename: str, company: str = "Unknown", year: str = "Unknown"):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents (doc_id, filename, company_ticker, filing_year)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, filename, company, year))
            conn.commit()

    def add_chunks(self, chunks_data: List[Dict]):
        """
        chunks_data format: [{'chunk_id': '..', 'doc_id': '..', 'section_name': '..', 'chunk_text': '..', 'token_count': ...}]
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO chunks (chunk_id, doc_id, section_name, chunk_text, token_count)
                VALUES (:chunk_id, :doc_id, :section_name, :chunk_text, :token_count)
            ''', chunks_data)
            conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM chunks WHERE chunk_id = ?', (chunk_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict]:
        if not chunk_ids:
            return []
        
        placeholders = ','.join('?' for _ in chunk_ids)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM chunks WHERE chunk_id IN ({placeholders})', chunk_ids)
            rows = cursor.fetchall()
            # Order them as requested
            chunk_map = {row['chunk_id']: dict(row) for row in rows}
            return [chunk_map[cid] for cid in chunk_ids if cid in chunk_map]

    def get_all_documents(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM documents')
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, doc_id: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM chunks WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
            conn.commit()
