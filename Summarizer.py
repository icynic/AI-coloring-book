"""Grounded biography summarization with Qwen3.5."""

from __future__ import annotations

import gc
import json
import re
import time

import torch
from transformers import BitsAndBytesConfig, pipeline


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class Summarizer:
    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        device_map="auto",
        quantization="none",
        dtype=None,
        revision=DEFAULT_MODEL_REVISION,
    ):
        self.model_name = model_name
        self.revision = revision
        self.quantization = quantization
        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32

        pipe_kwargs = {
            "task": "image-text-to-text",
            "model": model_name,
            "dtype": dtype,
            "device_map": device_map,
        }
        if revision:
            pipe_kwargs["revision"] = revision
        quantization_config = self._quantization_config(quantization)
        if quantization_config is not None:
            pipe_kwargs["model_kwargs"] = {"quantization_config": quantization_config}

        print(f"Loading summarizer {model_name} ({quantization})...")
        load_started_at = time.perf_counter()
        self.pipe = pipeline(**pipe_kwargs)
        self.model_load_seconds = round(time.perf_counter() - load_started_at, 3)

    @staticmethod
    def _quantization_config(quantization):
        if quantization in (None, "none"):
            return None
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16,
            )
        raise ValueError("quantization must be one of: none, 8bit, 4bit")

    @staticmethod
    def split_source_sentences(text):
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        # A lightweight splitter keeps this project self-contained. The complete
        # source text is retained, so sentence IDs can always be audited manually.
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned)
            if sentence.strip()
        ]

    @staticmethod
    def _extract_text(output):
        generated = output[0].get("generated_text", output[0])
        if isinstance(generated, str):
            return generated
        if isinstance(generated, list) and generated:
            content = generated[-1].get("content", generated[-1])
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
        return str(generated)

    @staticmethod
    def _parse_json_response(raw_text):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response did not contain a JSON object.")
        parsed = json.loads(raw_text[start : end + 1])
        summary = str(parsed.get("summary", "")).strip()
        if not summary:
            raise ValueError("The model response did not contain a summary.")
        evidence = parsed.get("supporting_source_sentence_ids", [])
        evidence = sorted({int(item) for item in evidence if str(item).isdigit()})
        return summary, evidence

    def summarize_with_evidence(
        self,
        text,
        target_age="10-14",
        min_words=80,
        max_words=110,
        max_new_tokens=384,
    ):
        sentences = self.split_source_sentences(text)
        if not sentences:
            raise ValueError("Cannot summarize empty source text.")

        numbered_source = "\n".join(
            f"[{index}] {sentence}" for index, sentence in enumerate(sentences, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You create concise educational biographies for children's coloring books. "
                            "Use only facts explicitly supported by the supplied source. Do not guess, "
                            "infer, embellish, or add outside knowledge. Return only valid JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Write a {min_words}-{max_words} word English biography for readers aged "
                            f"{target_age}. Use 4-6 clear sentences. Prefer important achievements and "
                            "avoid distressing or unnecessary detail. Return exactly this schema:\n"
                            '{"summary": "...", "supporting_source_sentence_ids": [1, 2]}\n\n'
                            "Every factual statement in the summary must be supported by at least one "
                            "listed source sentence.\n\nSOURCE:\n"
                            f"{numbered_source}"
                        ),
                    }
                ],
            },
        ]

        output = self.pipe(
            text=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        raw_text = self._extract_text(output)
        summary, evidence = self._parse_json_response(raw_text)
        valid_evidence = [index for index in evidence if 1 <= index <= len(sentences)]
        return {
            "summary": summary,
            "supporting_source_sentence_ids": valid_evidence,
            "supporting_source_sentences": [sentences[index - 1] for index in valid_evidence],
            "word_count": len(summary.split()),
            "target_age": target_age,
            "requested_word_range": [min_words, max_words],
            "model_id": self.model_name,
            "model_revision": self.revision,
            "quantization": self.quantization,
            "model_load_seconds": self.model_load_seconds,
            "raw_model_response": raw_text,
        }

    def summarize(self, text, max_new_tokens=384):
        """Backward-compatible helper returning only the biography text."""
        return self.summarize_with_evidence(text, max_new_tokens=max_new_tokens)["summary"]

    def cleanup(self):
        if hasattr(self, "pipe"):
            del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    example = (
        "Marie Curie was a Polish and naturalised-French physicist and chemist. "
        "She conducted pioneering research on radioactivity. She won Nobel Prizes "
        "in Physics and Chemistry and discovered polonium and radium."
    )
    summarizer = Summarizer()
    try:
        print(json.dumps(summarizer.summarize_with_evidence(example), indent=2))
    finally:
        summarizer.cleanup()
