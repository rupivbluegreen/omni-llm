"""Chat-aware text generation with multi-turn support."""

from __future__ import annotations
from typing import Generator

from tokenizer.special_tokens import SPECIAL_TOKENS, get_token_id
from tokenizer.chat_template import format_chat
from .prompts import DEFAULT_SYSTEM_PROMPT


class ChatGenerator:
    """Role-aware chat generation with thinking token support."""

    def __init__(self, model, tokenizer, max_ctx: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.max_ctx = max_ctx

        # Pre-resolve stop and control token IDs at init (not per-request)
        self.stop_token_ids = {
            get_token_id(tokenizer, "IM_END"),
            get_token_id(tokenizer, "EOS"),
            get_token_id(tokenizer, "IM_START"),  # Prevent model from role-playing user
        }
        self.think_start_id = get_token_id(tokenizer, "THINK_START")
        self.think_end_id = get_token_id(tokenizer, "THINK_END")

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.95,
        show_thinking: bool = False,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response for a chat conversation.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            show_thinking: If True, include thinking tokens in output
            system_prompt: Override default system prompt

        Yields:
            Generated text chunks
        """
        # Inject system prompt if not present
        if not any(m["role"] == "system" for m in messages):
            sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            messages = [{"role": "system", "content": sys_prompt}] + messages

        # Truncate history to fit context budget
        generation_budget = max_tokens + 50  # margin for special tokens
        max_prompt_tokens = self.max_ctx - generation_budget
        messages = self._truncate_history(messages, max_prompt_tokens)

        # Format and tokenize
        prompt_text = format_chat(messages, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(prompt_text).ids

        # Generate tokens
        in_thinking = False
        generated = 0

        for token_id in self._sample(prompt_ids, max_tokens, temperature, top_p):
            if token_id in self.stop_token_ids:
                break

            generated += 1

            # Handle thinking tokens by ID comparison
            if token_id == self.think_start_id:
                in_thinking = True
                if show_thinking:
                    yield "<think>\n"
                continue
            if token_id == self.think_end_id:
                in_thinking = False
                if show_thinking:
                    yield "\n</think>\n"
                continue

            if in_thinking and not show_thinking:
                continue

            yield self.tokenizer.decode([token_id])

    def _truncate_history(
        self, messages: list[dict], max_tokens: int
    ) -> list[dict]:
        """Keep system prompt + most recent turns that fit within token budget.
        Uses actual tokenization for counting (not heuristic)."""
        if not messages:
            return messages

        # Tokenize each message to get actual token counts
        from tokenizer.chat_template import format_message

        msg_tokens = []
        for msg in messages:
            text = format_message(msg["role"], msg["content"])
            count = len(self.tokenizer.encode(text).ids)
            msg_tokens.append(count)

        total = sum(msg_tokens)
        if total <= max_tokens:
            return messages

        # Always keep system message (index 0) and latest user message (last)
        kept = []
        budget = max_tokens

        # Reserve space for system (first) and current exchange (last 1-2 messages)
        system_msgs = [i for i, m in enumerate(messages) if m["role"] == "system"]
        if system_msgs:
            sys_idx = system_msgs[0]
            kept.append(sys_idx)
            budget -= msg_tokens[sys_idx]

        # Add messages from newest to oldest (skip system)
        non_system = [i for i in range(len(messages)) if i not in kept]
        selected_tail = []

        for i in reversed(non_system):
            if budget - msg_tokens[i] < 0:
                break
            selected_tail.insert(0, i)
            budget -= msg_tokens[i]

        # Combine: system + selected recent messages
        all_indices = sorted(set(kept + selected_tail))
        return [messages[i] for i in all_indices]

    def _sample(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[int, None, None]:
        """Sample tokens from the model autoregressively.
        This is a framework-agnostic interface — actual implementation
        depends on whether MLX or another backend is used."""
        try:
            import mlx.core as mx
        except ImportError:
            # Fallback for non-MLX environments (testing/dev)
            return

        tokens = mx.array([prompt_ids])

        for _ in range(max_tokens):
            logits = self.model(tokens)
            next_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = mx.argmax(next_logits, axis=-1)
            else:
                next_logits = next_logits / temperature
                # Top-p sampling
                if top_p < 1.0:
                    sorted_indices = mx.argsort(-next_logits, axis=-1)
                    sorted_logits = mx.take_along_axis(next_logits, sorted_indices, axis=-1)
                    probs = mx.softmax(sorted_logits, axis=-1)
                    cumprobs = mx.cumsum(probs, axis=-1)
                    mask = cumprobs - probs > top_p
                    sorted_logits = mx.where(mask, -float("inf"), sorted_logits)
                    # Unsort
                    next_logits = mx.zeros_like(next_logits)
                    next_logits = next_logits.at[mx.arange(sorted_indices.shape[-1])].set(sorted_logits.squeeze(0))

                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))

            token_id = next_token.item()
            yield token_id

            tokens = mx.concatenate([tokens, next_token.reshape(1, 1)], axis=1)
