import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class OmniscientConfig:
    vocab_size: int = 32_768
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4         # GQA: 4 KV heads shared across 16 query heads
    ffn_dim: int = 2816         # SwiGLU intermediate size
    max_seq_len: int = 4096
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-6
    weight_tying: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OmniscientConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "OmniscientConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
