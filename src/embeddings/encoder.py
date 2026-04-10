from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from src import config

class DocumentEncoder:
    """
    Handles encoding of chunks and queries into dense vectors using a base model 
    (default: BAAI/bge-base-en-v1.5).
    """
    def __init__(self, model_name: str = config.BASE_ENCODER_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        # BGE requires queries to have a specific prefix for best performance
        self.is_bge = "bge" in model_name.lower()

    def encode_queries(self, queries: Union[str, List[str]]) -> torch.Tensor:
        """
        Embed queries. BGE base requires 'Represent this sentence for searching relevant passages: ' prefix.
        """
        if isinstance(queries, str):
            queries = [queries]
            
        if self.is_bge:
            queries = [f"Represent this sentence for searching relevant passages: {q}" for q in queries]
            
        embeddings = self.model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings

    def encode_documents(self, documents: Union[str, List[str]]) -> torch.Tensor:
        """
        Embed document chunks. BGE does not require a prefix for documents.
        """
        if isinstance(documents, str):
            documents = [documents]
            
        # normalize_embeddings=True is recommended for BGE to use cosine similarity via dot product
        embeddings = self.model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
        return embeddings
