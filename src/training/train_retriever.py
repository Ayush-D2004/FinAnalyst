"""
train_retriever.py
──────────────────
Two-stage retriever fine-tuning pipeline for the SEC RAG assistant.

Stage 1 – Domain Adaptation  (unsupervised, --stage domain)
  Uses TSDAE (Transformer-based Sequential Denoising Auto-Encoder)
  on raw SEC 10-K chunks. No labels required.
  → The model learns financial vocabulary, abbreviations, and document layout.

Stage 2 – Contrastive Search  (supervised, --stage contrastive)
  Uses MultipleNegativesRankingLoss on the triplet CSV produced by
  FinanceBench_EDGAR_Mapping.py.
  → The model learns which chunks actually answer a given query.

Both stages export portable model weights that can be loaded by the
Streamlit app by pointing config.BASE_ENCODER_MODEL at the output path.

Designed to run on Google Colab (T4/A100) or Kaggle.
CPU execution is possible but slow for Stage 1.

Usage (in Colab):
    # Stage 1 – feed it the raw SEC chunks txt file
    !python src/training/train_retriever.py \\
        --stage domain \\
        --data data/training/raw_chunks.txt \\
        --output models/fin-bge-stage1

    # Stage 2 – feed it the mined triplet CSV
    !python src/training/train_retriever.py \\
        --stage contrastive \\
        --base  models/fin-bge-stage1 \\
        --data  data/training/triplet_train_edgar.csv \\
        --output models/fin-bge-stage2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    datasets as st_datasets,
    losses,
)

from src import config


# ── Stage 1: Domain Adaptation via TSDAE ─────────────────────────────────────

def train_domain_adaptation(
    raw_text_path: str,
    base_model: str,
    output_path: str,
    epochs: int = 1,
    batch_size: int = 8,
) -> None:
    """
    TSDAE – trains by deleting 60 % of tokens from a chunk and asking the
    model to reconstruct the original sentence, forcing it to learn the
    dense semantics of SEC financial language.

    Args:
        raw_text_path:  Path to a newline-separated .txt file of raw SEC
                        chunk texts (one chunk per line).
    """
    print(f"\n{'='*60}")
    print("  Stage 1: Domain Adaptation (TSDAE)")
    print(f"  Base model : {base_model}")
    print(f"  Data file  : {raw_text_path}")
    print(f"  Output     : {output_path}")
    print(f"{'='*60}\n")

    model = SentenceTransformer(base_model)

    # Build DenoisingAutoEncoderDataset
    with open(raw_text_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if len(line.strip().split()) > 10]

    print(f"  Loaded {len(sentences)} raw SEC sentences for TSDAE.")

    train_dataset = st_datasets.DenoisingAutoEncoderDataset(sentences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    tsdae_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=base_model,
        tie_encoder_decoder=True,
    )

    warmup_steps = int(len(train_loader) * epochs * 0.1)

    model.fit(
        train_objectives=[(train_loader, tsdae_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_save_steps=500,
        checkpoint_path=str(Path(output_path) / "checkpoints"),
    )

    print(f"\n  Domain-adapted model written to: {output_path}")


# ── Stage 2: Supervised Contrastive Fine-Tuning ───────────────────────────────

def train_contrastive(
    triplet_csv: str,
    base_model: str,
    output_path: str,
    epochs: int = 3,
    batch_size: int = 16,
) -> None:
    """
    MultipleNegativesRankingLoss fine-tuning using (query, positive, hard_neg)
    triplets mined from EDGAR + FinanceBench.
    """
    print(f"\n{'='*60}")
    print("  Stage 2: Contrastive Fine-Tuning (MNRL)")
    print(f"  Base model : {base_model}")
    print(f"  Data file  : {triplet_csv}")
    print(f"  Output     : {output_path}")
    print(f"{'='*60}\n")

    df = pd.read_csv(triplet_csv)
    required = {"query", "positive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"triplet CSV is missing columns: {missing}. "
            "Run FinanceBench_EDGAR_Mapping.py first."
        )

    model = SentenceTransformer(base_model)

    examples = []
    for _, row in df.iterrows():
        q  = str(row["query"]).strip()
        p  = str(row["positive"]).strip()
        hn = str(row.get("hard_negative", "")).strip()
        if not q or not p:
            continue
        # With MNRL, include hard negative as a third text when available
        texts = [q, p, hn] if hn else [q, p]
        examples.append(InputExample(texts=texts))

    print(f"  Training examples: {len(examples)}")

    train_loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = max(1, int(len(train_loader) * epochs * 0.1))

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_save_steps=500,
        checkpoint_path=str(Path(output_path) / "checkpoints"),
    )

    print(f"\n  Fine-tuned model written to: {output_path}")
    print("  Load it back with:")
    print(f'    BASE_ENCODER_MODEL = "{output_path}"  # in src/config.py')


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the SEC financial retriever.")
    p.add_argument(
        "--stage", required=True, choices=["domain", "contrastive"],
        help="Which training stage to run.",
    )
    p.add_argument(
        "--data", required=True,
        help=(
            "For 'domain'      → path to newline-separated raw chunk .txt file.\n"
            "For 'contrastive' → path to triplet CSV with [query, positive, hard_negative]."
        ),
    )
    p.add_argument(
        "--base",
        default=config.BASE_ENCODER_MODEL,
        help="Base model name or local checkpoint path (default: config.BASE_ENCODER_MODEL).",
    )
    p.add_argument(
        "--output", required=True,
        help="Path where the fine-tuned model will be saved.",
    )
    p.add_argument("--epochs",     type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.stage == "domain":
        train_domain_adaptation(
            raw_text_path=args.data,
            base_model=args.base,
            output_path=args.output,
            epochs=args.epochs or 1,
            batch_size=args.batch_size or 8,
        )
    else:
        train_contrastive(
            triplet_csv=args.data,
            base_model=args.base,
            output_path=args.output,
            epochs=args.epochs or 3,
            batch_size=args.batch_size or 16,
        )
