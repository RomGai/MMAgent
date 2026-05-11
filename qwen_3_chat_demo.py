"""Callable Qwen3 chat wrapper used by prompt-replaceable modules.

The file can still be executed as a demo, but importing it no longer loads the
model immediately.  This makes it safe for unit tests and for deterministic
rule-based modules that only need an optional LLM client interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class QwenChatClient:
    model_name: str = "Qwen/Qwen3-8B"
    torch_dtype: str = "auto"
    device_map: str = "auto"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
        return_thinking: bool = False,
    ) -> str | dict[str, str]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        think_token_id = 151668
        index = 0
        if think_token_id in output_ids:
            index = len(output_ids) - output_ids[::-1].index(think_token_id)

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        if return_thinking:
            return {"thinking_content": thinking_content, "content": content}
        return content


def chat_once(prompt: str, *, enable_thinking: bool = False, max_new_tokens: int = 2048) -> str:
    client = QwenChatClient()
    return client.chat(
        [{"role": "user", "content": prompt}],
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
    )  # type: ignore[return-value]


if __name__ == "__main__":
    print(chat_once("Give me a short introduction to large language model."))
