"""Tests for SFT data formatting and assistant-only loss masking."""

from unittest.mock import MagicMock
from data.sft_data import tokenize_conversation, SFTExample, load_sft_dataset
import tempfile
import json


def make_mock_tokenizer():
    """Create a mock tokenizer that returns predictable token IDs."""
    tokenizer = MagicMock()

    # Simple encoding: each character becomes a token
    def mock_encode(text):
        result = MagicMock()
        result.ids = list(range(len(text)))
        return result

    tokenizer.encode = mock_encode
    tokenizer.token_to_id.return_value = 0
    return tokenizer


class TestTokenizeConversation:
    def test_returns_sft_example(self):
        tokenizer = make_mock_tokenizer()
        conv = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = tokenize_conversation(conv, tokenizer)
        assert isinstance(result, SFTExample)
        assert len(result.input_ids) == len(result.labels)

    def test_system_tokens_masked(self):
        tokenizer = make_mock_tokenizer()
        conv = [{"role": "system", "content": "system prompt"}]
        result = tokenize_conversation(conv, tokenizer)
        assert all(label == -100 for label in result.labels)

    def test_user_tokens_masked(self):
        tokenizer = make_mock_tokenizer()
        conv = [{"role": "user", "content": "user message"}]
        result = tokenize_conversation(conv, tokenizer)
        assert all(label == -100 for label in result.labels)

    def test_assistant_has_unmasked_tokens(self):
        tokenizer = make_mock_tokenizer()
        conv = [{"role": "assistant", "content": "response"}]
        result = tokenize_conversation(conv, tokenizer)
        # Some tokens should NOT be -100 (the assistant content)
        assert any(label != -100 for label in result.labels)

    def test_truncation(self):
        tokenizer = make_mock_tokenizer()
        conv = [{"role": "user", "content": "x" * 10000}]
        result = tokenize_conversation(conv, tokenizer, max_seq_len=100)
        assert len(result.input_ids) <= 100
        assert len(result.labels) <= 100


class TestLoadSFTDataset:
    def test_load_from_jsonl(self):
        data = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            f.flush()
            result = load_sft_dataset(f.name)

        assert len(result) == 2
        assert result[0][0]["role"] == "user"
        assert result[1][1]["content"] == "goodbye"
