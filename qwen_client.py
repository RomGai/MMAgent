"""Reusable Qwen3-8B chat client.

This module wraps the inference pattern from ``qwen_3_chat_demo.py`` behind a
small ``chat(messages) -> str`` interface so the same loaded Qwen3-8B backbone
can be reused across multiple profile-rewriting calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QwenGenerationConfig:
    """Generation options for Qwen3-8B."""

    model_name: str = "Qwen/Qwen3-8B"
    max_new_tokens: int = 32768
    temperature: float | None = 0.2
    top_p: float | None = 0.9
    do_sample: bool = False
    enable_thinking: bool = False


class QwenChatClient:
    """Lazy, reusable chat client backed by Qwen3-8B.

    The heavy tokenizer/model objects are loaded on the first ``chat`` call,
    not at import time. This keeps CLI help, unit tests, and parser checks fast
    while still guaranteeing that real inference uses the required Qwen3-8B
    backbone.
    """

    def __init__(self, config: QwenGenerationConfig | None = None) -> None:
        self.config = config or QwenGenerationConfig()
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self.tokenizer is not None and self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Run one chat completion and return only the final answer text.

        Args:
            messages: OpenAI-style chat messages, e.g.
                ``[{"role": "user", "content": "..."}]``.

        Returns:
            The decoded assistant content after any Qwen thinking segment.
        """

        self._ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
        }
        if self.config.do_sample:
            generation_kwargs["do_sample"] = True
            if self.config.temperature is not None:
                generation_kwargs["temperature"] = self.config.temperature
            if self.config.top_p is not None:
                generation_kwargs["top_p"] = self.config.top_p
        else:
            generation_kwargs["do_sample"] = False

        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # Qwen3 uses token id 151668 for </think>. Keep the demo's extraction
        # behavior: when thinking exists, return content after the final marker.
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        return self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


def chat(messages: list[dict[str, str]]) -> str:
    """Convenience one-shot chat function using a module-level Qwen3-8B client."""

    global _DEFAULT_CLIENT
    try:
        client = _DEFAULT_CLIENT
    except NameError:
        client = _DEFAULT_CLIENT = QwenChatClient()
    return client.chat(messages)
