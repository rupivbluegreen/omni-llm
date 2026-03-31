"""Fill-in-the-Middle completion generator."""

from __future__ import annotations
from typing import Generator

from tokenizer.special_tokens import SPECIAL_TOKENS, get_token_id


class FIMGenerator:
    """Generate code completions using Fill-in-the-Middle format."""

    def __init__(self, model, tokenizer, max_ctx: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.max_ctx = max_ctx

        # Pre-resolve token IDs
        self.fim_middle_id = get_token_id(tokenizer, "FIM_MIDDLE")
        self.eos_id = get_token_id(tokenizer, "EOS")

        self.stop_ids = {self.eos_id, self.fim_middle_id}

    def complete(
        self,
        prefix: str,
        suffix: str = "",
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """Generate a FIM completion.

        Args:
            prefix: Code before the cursor
            suffix: Code after the cursor
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            The generated middle text
        """
        # Format as FIM: <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>
        fim_text = (
            SPECIAL_TOKENS["FIM_PREFIX"] + prefix
            + SPECIAL_TOKENS["FIM_SUFFIX"] + suffix
            + SPECIAL_TOKENS["FIM_MIDDLE"]
        )
        prompt_ids = self.tokenizer.encode(fim_text).ids

        # Truncate prefix if too long (keep suffix intact)
        max_prompt = self.max_ctx - max_tokens
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        # Generate
        output_ids = []
        for token_id in self._sample(prompt_ids, max_tokens, temperature):
            if token_id in self.stop_ids:
                break
            output_ids.append(token_id)

        return self.tokenizer.decode(output_ids)

    def _sample(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
    ) -> Generator[int, None, None]:
        """Sample tokens autoregressively."""
        try:
            import mlx.core as mx
        except ImportError:
            return

        tokens = mx.array([prompt_ids])

        for _ in range(max_tokens):
            logits = self.model(tokens)
            next_logits = logits[:, -1, :]

            if temperature == 0:
                next_token = mx.argmax(next_logits, axis=-1)
            else:
                next_logits = next_logits / temperature
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))

            token_id = next_token.item()
            yield token_id

            tokens = mx.concatenate([tokens, next_token.reshape(1, 1)], axis=1)
