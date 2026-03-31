import mlx.core as mx
import mlx.nn as nn

from .config import OmniscientConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class Attention(nn.Module):
    def __init__(self, config: OmniscientConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups
        self.scale = config.head_dim ** -0.5

        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape

        q = self.wq(x)  # (B, L, n_heads * head_dim)
        k = self.wk(x)  # (B, L, n_kv_heads * head_dim)
        v = self.wv(x)  # (B, L, n_kv_heads * head_dim)

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        # GQA: expand KV heads to match query heads
        k = mx.repeat(k, self.n_kv_groups, axis=1)  # (B, n_heads, L, head_dim)
        v = mx.repeat(v, self.n_kv_groups, axis=1)  # (B, n_heads, L, head_dim)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, n_heads, L, L)
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)

        out = weights @ v  # (B, n_heads, L, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, config: OmniscientConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.ffn_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))
