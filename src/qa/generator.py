import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict
from src import config

class QAGenerator:
    """
    Handles prompt construction and model generation ensuring strict grounding.
    Explicitly instructs the model to refuse if evidence is missing.
    Uses model.generate() directly to avoid pipeline task-name changes across
    transformers versions (text2text-generation was removed in v5).
    """
    def __init__(self, model_name: str = config.GENERATOR_MODEL):
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.torch_device)

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generates an answer strictly grounded in the provided context_chunks.
        """
        if not context_chunks:
            return "I cannot answer this question because the document does not contain relevant information."

        # Build context, limiting the total length so we don't truncate the prompt query
        # 512 tokens is roughly 2000 characters. We leave room for instructions and query.
        max_context_chars = 1500
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            addition = f"\n[Doc {i+1}] {chunk['chunk_text']}"
            if len(context_text) + len(addition) > max_context_chars:
                # Add as much as possible then break
                remaining = max_context_chars - len(context_text)
                if remaining > 50:
                    context_text += addition[:remaining] + "...[TRUNCATED]"
                break
            context_text += addition
            
        # Instruction prompt engineered for strict grounding
        # Put question before context to ensure it's not truncated by tokenizer
        prompt = (
            "You are a strict financial analyst assistant. Use ONLY the provided document context to answer the question. "
            "If the context does not contain enough information, respond exactly with: "
            "'I cannot answer this question because the document does not contain relevant information.' "
            "Cite the document numbers e.g. [Doc 1].\n\n"
            f"Question: {query}\n\n"
            f"Context:{context_text}\n\n"
            "Answer:"
        )
        
        # Generator params
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.torch_device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
