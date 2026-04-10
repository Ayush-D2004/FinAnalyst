import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.embeddings.encoder import DocumentEncoder
from src import config

class Evaluator:
    """
    Deeper evaluation metrics comparing Baseline vs Fine-tuned vs Reranked.
    """
    def __init__(self, eval_dataset_path: str):
        # CSV expecting: [query, relevant_doc_id]
        self.df = pd.read_csv(eval_dataset_path)
        
    def calculate_mrr(self, predictions: List[List[str]], ground_truths: List[str]) -> float:
        mrr = 0.0
        for preds, gt in zip(predictions, ground_truths):
            try:
                rank = preds.index(gt) + 1
                mrr += 1.0 / rank
            except ValueError:
                pass
        return mrr / max(len(ground_truths), 1)
        
    def calculate_recall_at_k(self, predictions: List[List[str]], ground_truths: List[str], k: int) -> float:
        hits = sum(1 for preds, gt in zip(predictions, ground_truths) if gt in preds[:k])
        return hits / max(len(ground_truths), 1)

    def run_ablation_study(self, base_encoder: SentenceTransformer, ft_encoder: SentenceTransformer, reranker: CrossEncoder, faiss_docs: Dict[str, str]):
        """
        Runs baseline, dense finetuned, and reranking across queries to produce comparative metrics.
        """
        queries = self.df['query'].tolist()
        gts = self.df['relevant_doc_id'].tolist()
        
        # NOTE: Full indexing step omitted for brevity in evaluator stub.
        # This function would encode `faiss_docs` with both base & ft encoders,
        # then calculate MRR/Recall locally.
        print("Ablation Study Framework initialized. Ready to execute on full dataset...")
        
    def answer_faithfulness_check(self, answer: str, context: str) -> float:
        """
        Simple heuristic logic for checking faithfulness bound.
        Returns score [0.0 - 1.0].
        """
        # Exact match / precision logic (In production, replace with NLI Model / LLM Judge)
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words: return 0.0
        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

if __name__ == "__main__":
    print("Evaluator Module Loaded.")
