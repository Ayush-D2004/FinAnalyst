import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple
from src import config

class FAISSStore:
    """
    Manages the FAISS index and local disk persistence.
    Works closely with SQLiteStore. Since SQLite uses string UUIDs and FAISS uses int64 IDs,
    we maintain a mapping (or just rely on order if using a flat index with simple add).
    To map UUIDs to FAISS IDs, we can use IndexIDMap or a separate dict persisted to disk.
    For simplicity and robustness, we will use a separate mapping dict saved alongside the index.
    """
    def __init__(self, index_path: Path = config.FAISS_INDEX_PATH, dim: int = config.EXPECTED_EMBEDDING_DIM):
        self.index_path = index_path
        self.dim = dim
        self.chunk_ids: List[str] = []  # FAISS internal index 0...N maps to chunk_ids[0...N]
        
        self.mapping_path = index_path.with_suffix('.ids')
        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.mapping_path.exists():
                with open(self.mapping_path, 'r') as f:
                    self.chunk_ids = f.read().splitlines()
        else:
            # We use Inner Product because encoder normalizes embeddings (Cosine Sim = IP)
            self.index = faiss.IndexFlatIP(self.dim)

    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str]):
        if len(embeddings) != len(chunk_ids):
            raise ValueError("Embeddings and chunk_ids must have the same length")
        
        # Ensure numpy array type
        if not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.cpu().numpy()
            
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Search the index and return list of (chunk_id, score).
        """
        if self.index.ntotal == 0:
            return []
            
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.cpu().numpy()
            
        # Ensure 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        scores, I = self.index.search(query_embedding, top_k)
        
        results = []
        for j in range(len(I[0])):
            idx = I[0][j]
            if idx != -1 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(scores[0][j])))
                
        return results

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, 'w') as f:
            f.write('\n'.join(self.chunk_ids))
