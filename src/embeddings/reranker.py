import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple
from src import config

class DocumentReranker:
    """
    Uses a Cross-Encoder to rerank a list of retrieved chunks against a query.
    This provides higher precision than the base bi-encoder.
    """
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CrossEncoder returns single logits. Default is usually max_length=512
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)

    def rerank(self, query: str, chunks: List[str], top_k: int = config.RERANK_TOP_K) -> List[Tuple[int, float]]:
        """
        Reranks the chunks against the query.
        Returns a list of tuples: (original_index, score) sorted by score descending.
        """
        if not chunks:
            return []
            
        # CrossEncoder expects pairs of [query, chunk]
        pairs = [[query, chunk] for chunk in chunks]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        scored_pairs = [(i, float(score)) for i, score in enumerate(scores)]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_pairs[:top_k]
