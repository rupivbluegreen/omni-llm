"""Tests for model configuration."""

import json
import tempfile
from pathlib import Path

from model.config import OmniscientConfig


class TestOmniscientConfig:
    def test_defaults(self):
        config = OmniscientConfig()
        assert config.vocab_size == 32_768
        assert config.d_model == 1024
        assert config.n_layers == 16
        assert config.n_heads == 16
        assert config.n_kv_heads == 4
        assert config.ffn_dim == 2816
        assert config.max_seq_len == 4096
        assert config.rope_theta == 500_000.0
        assert config.rms_norm_eps == 1e-6
        assert config.weight_tying is True

    def test_head_dim(self):
        config = OmniscientConfig()
        assert config.head_dim == 64  # 1024 / 16

    def test_n_kv_groups(self):
        config = OmniscientConfig()
        assert config.n_kv_groups == 4  # 16 / 4

    def test_to_dict_roundtrip(self):
        config = OmniscientConfig()
        d = config.to_dict()
        restored = OmniscientConfig.from_dict(d)
        assert config == restored

    def test_from_dict_ignores_extra_keys(self):
        d = OmniscientConfig().to_dict()
        d["unknown_key"] = "should be ignored"
        config = OmniscientConfig.from_dict(d)
        assert config.d_model == 1024

    def test_save_load(self):
        config = OmniscientConfig(d_model=512, n_layers=8)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config.save(f.name)
            loaded = OmniscientConfig.load(f.name)
        assert loaded.d_model == 512
        assert loaded.n_layers == 8
        Path(f.name).unlink()
