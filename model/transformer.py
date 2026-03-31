import mlx.core as mx
import mlx.nn as nn

from .config import OmniscientConfig
from .layers import RMSNorm, Attention, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: OmniscientConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attention(self.attention_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OmniscientModel(nn.Module):
    def __init__(self, config: OmniscientConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.output.weight = self.tok_embeddings.weight

    def __call__(self, tokens: mx.array, mask: mx.array | None = None) -> mx.array:
        h = self.tok_embeddings(tokens)

        if mask is None:
            _, L = tokens.shape
            mask = mx.full((L, L), float("-inf"))
            mask = mx.triu(mask, k=1)

        for layer in self.layers:
            h = layer(h, mask=mask)

        h = self.norm(h)
        return self.output(h)
