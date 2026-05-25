"""
generator.py
─────────────
Dual-backend answer generator for SEC Document Intelligence.

Supports two backends:
  1. **Gemini** (default) – Uses the google-genai SDK for high-quality
     financial QA with large context windows.
  2. **Flan-T5-Base** (local) – Runs entirely on-device for offline use.

Both backends receive the same structured prompt and return grounded,
citation-tagged answers.
"""
import re
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from src import config


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class BaseGenerator(ABC):
    """Common interface for answer generators."""

    SYSTEM_PROMPT = (
        "You are a strict financial analyst assistant. "
        "Use ONLY the provided document context to answer the question. "
        "Answer in 2 to 4 concise sentences. Start with the direct answer "
        "(include specific numbers, dollar amounts, or percentages when available), "
        "then give brief supporting evidence from the document. "
        "Cite the document numbers in square brackets, e.g. [Doc 1]. "
        "If the context does not contain enough information to answer, respond exactly with: "
        "'I cannot answer this question because the document does not contain relevant information.'"
    )

    def _build_context(self, context_chunks: List[Dict], max_chars: int = 6000) -> str:
        """Build a context string from retrieved chunks, respecting a character budget."""
        context_parts = []
        total_len = 0
        for i, chunk in enumerate(context_chunks):
            text = chunk.get("chunk_text", "")
            addition = f"\n[Doc {i+1}] {text}"
            if total_len + len(addition) > max_chars:
                remaining = max_chars - total_len
                if remaining > 80:
                    context_parts.append(addition[:remaining] + "...[TRUNCATED]")
                break
            context_parts.append(addition)
            total_len += len(addition)
        return "".join(context_parts)

    def _build_prompt(self, query: str, context_text: str) -> str:
        """Build the user prompt (system prompt handled separately where possible)."""
        return (
            f"Question: {query}\n\n"
            f"Context:{context_text}\n\n"
            "Answer:"
        )

    @abstractmethod
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini backend
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiGenerator(BaseGenerator):
    """Answer generator using the Google Gemini API via google-genai."""

    def __init__(self, model_name: str = config.GEMINI_MODEL, api_key: str = config.GEMINI_API_KEY):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def get_name(self) -> str:
        return f"Gemini ({self._model_name})"

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I cannot answer this question because the document does not contain relevant information."

        from google.genai import types
        import time

        context_text = self._build_context(context_chunks, max_chars=12000)
        user_prompt = self._build_prompt(query, context_text)

        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.SYSTEM_PROMPT,
                        temperature=0.2,
                        max_output_tokens=300,
                    ),
                )
                answer = response.text.strip()
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = "429" in exc_str or "resource_exhausted" in exc_str or "quota" in exc_str
                if is_rate_limit and attempt < max_retries:
                    sleep_time = backoff * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                return f"Gemini API error after {attempt} retries: {exc}"

        if answer.startswith("Answer:"):
            answer = answer[len("Answer:"):].strip()
        return answer


# ═══════════════════════════════════════════════════════════════════════════════
# Flan-T5 local backend
# ═══════════════════════════════════════════════════════════════════════════════

class FlanT5Generator(BaseGenerator):
    """
    Local Flan-T5-Base generator. Improved but fundamentally limited by the
    250M parameter model and 512-token input window.
    """

    # Simplified direct prompt that the 250M model can follow successfully
    FLAN_SYSTEM_PROMPT = (
        "You are a financial analyst. Answer the question directly using the numbers from the context. "
        "If the context does not contain the answer, say exactly: "
        "'I cannot answer this question because the document does not contain relevant information.'"
    )

    def __init__(self, model_name: str = config.GENERATOR_MODEL):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._torch_device)

    def get_name(self) -> str:
        return "Flan-T5-Base (local)"

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        import torch

        if not context_chunks:
            return "I cannot answer this question because the document does not contain relevant information."

        # Build the preamble (system + question) first to measure its token cost
        preamble = self.FLAN_SYSTEM_PROMPT + f"\n\nQuestion: {query}\n\nContext:"
        preamble_tokens = len(self._tokenizer.encode(preamble, add_special_tokens=True))

        # Flan-T5-Base has a 512-token input limit
        max_input_tokens = 512
        available_for_context = max_input_tokens - preamble_tokens - 10  # safety margin

        # Build context that fits within the remaining token budget
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            addition = f"\n[Doc {i+1}] {chunk.get('chunk_text', '')}"
            if len(self._tokenizer.encode(context_text + addition, add_special_tokens=False)) > available_for_context:
                # Try to fit a truncated version
                remaining_tokens = available_for_context - len(
                    self._tokenizer.encode(context_text, add_special_tokens=False)
                )
                if remaining_tokens > 30:
                    # Truncate the addition by words
                    words = addition.split()
                    truncated = ""
                    for w in words:
                        candidate = truncated + " " + w if truncated else w
                        if len(self._tokenizer.encode(candidate, add_special_tokens=False)) > remaining_tokens:
                            break
                        truncated = candidate
                    if truncated:
                        context_text += truncated + "...[TRUNCATED]"
                break
            context_text += addition

        prompt = preamble + context_text + "\n\nAnswer:"

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_input_tokens,
            truncation=True,
        ).to(self._torch_device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=150,
            min_new_tokens=8,
            num_beams=4,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
        answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if answer.startswith("Answer:"):
            answer = answer[len("Answer:"):].strip()

        # Clean up: collapse excessive whitespace but do NOT merge numbers
        answer = re.sub(r"\s+", " ", answer).strip()

        return answer



# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

class QAGenerator:
    """
    Facade that wraps both backends and supports runtime switching.
    Lazily initialises backends on first use.
    """

    def __init__(self):
        self._backends: Dict[str, BaseGenerator] = {}
        self._active_backend: Optional[str] = None

    @property
    def available_backends(self) -> List[str]:
        backends = ["Flan-T5 (Local)"]
        if config.GEMINI_API_KEY:
            backends.insert(0, "Gemini (API)")
        return backends

    @property
    def active_backend_name(self) -> str:
        if self._active_backend:
            return self._active_backend
        # Default: Gemini if API key is set, otherwise Flan-T5
        if config.GEMINI_API_KEY:
            return "Gemini (API)"
        return "Flan-T5 (Local)"

    def set_backend(self, name: str):
        self._active_backend = name

    def _get_backend(self, name: str) -> BaseGenerator:
        if name not in self._backends:
            if name == "Gemini (API)":
                self._backends[name] = GeminiGenerator()
            elif name == "Flan-T5 (Local)":
                self._backends[name] = FlanT5Generator()
            else:
                raise ValueError(f"Unknown backend: {name}")
        return self._backends[name]

    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        backend_name = self.active_backend_name
        backend = self._get_backend(backend_name)
        return backend.generate_answer(query, context_chunks)

    def get_active_display_name(self) -> str:
        backend_name = self.active_backend_name
        try:
            backend = self._get_backend(backend_name)
            return backend.get_name()
        except Exception:
            return backend_name
