import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Set
from src import config

class FAISSStore:
    """
    Manages the FAISS index and local disk persistence.

    Supports per-document filtering: callers can pass a set of valid chunk_ids
    to restrict search results, avoiding cross-document contamination when
    multiple documents share the same index.
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

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        allowed_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search the index and return list of (chunk_id, score).

        Args:
            query_embedding: The query vector(s).
            top_k: Number of results to return.
            allowed_chunk_ids: If provided, only return results whose chunk_id
                               is in this set. We over-fetch from FAISS and
                               filter, ensuring we still return up to top_k
                               results for the target document.
        """
        if self.index.ntotal == 0:
            return []
            
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.cpu().numpy()
            
        # Ensure 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # If filtering, over-fetch to compensate for discarded results
        fetch_k = top_k
        if allowed_chunk_ids is not None:
            # Fetch more to ensure we get enough after filtering
            fetch_k = min(self.index.ntotal, top_k * 5)

        scores, I = self.index.search(query_embedding, fetch_k)
        
        results = []
        for j in range(len(I[0])):
            idx = I[0][j]
            if idx == -1 or idx >= len(self.chunk_ids):
                continue
            cid = self.chunk_ids[idx]
            # Apply document filter if provided
            if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                continue
            results.append((cid, float(scores[0][j])))
            if len(results) >= top_k:
                break
                
        return results

    def save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, 'w') as f:
            f.write('\n'.join(self.chunk_ids))

    def clear(self):
        """Remove all data from the index and mapping file."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunk_ids = []
        # Remove files from disk
        if self.index_path.exists():
            self.index_path.unlink()
        if self.mapping_path.exists():
            self.mapping_path.unlink()
