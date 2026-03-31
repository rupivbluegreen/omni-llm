"""Tests for tokenizer special tokens and chat template."""

from tokenizer.special_tokens import SPECIAL_TOKENS, SPECIAL_TOKEN_LIST
from tokenizer.chat_template import format_message, format_chat, format_chat_with_thinking


class TestSpecialTokens:
    def test_token_count(self):
        assert len(SPECIAL_TOKENS) == 18

    def test_token_list_matches_dict(self):
        assert len(SPECIAL_TOKEN_LIST) == len(SPECIAL_TOKENS)
        assert set(SPECIAL_TOKEN_LIST) == set(SPECIAL_TOKENS.values())

    def test_all_tokens_have_delimiters(self):
        for token in SPECIAL_TOKEN_LIST:
            assert token.startswith("<|"), f"{token} missing <| prefix"
            assert token.endswith("|>"), f"{token} missing |> suffix"

    def test_no_duplicate_tokens(self):
        assert len(SPECIAL_TOKEN_LIST) == len(set(SPECIAL_TOKEN_LIST))


class TestChatTemplate:
    def test_format_message(self):
        result = format_message("user", "hello")
        assert "<|im_start|>" in result
        assert "<|user|>" in result
        assert "hello" in result
        assert "<|im_end|>" in result

    def test_format_message_system(self):
        result = format_message("system", "you are helpful")
        assert "<|system|>" in result
        assert "you are helpful" in result

    def test_format_chat_single_turn(self):
        messages = [{"role": "user", "content": "hi"}]
        result = format_chat(messages, add_generation_prompt=True)
        assert "<|user|>" in result
        assert "hi" in result
        # Should end with assistant prompt
        assert result.rstrip().endswith("<|assistant|>") or "<|assistant|>" in result

    def test_format_chat_multi_turn(self):
        messages = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
        result = format_chat(messages)
        assert result.index("<|system|>") < result.index("<|user|>")
        assert result.count("<|im_start|>") == 5  # 4 messages + generation prompt

    def test_format_chat_no_generation_prompt(self):
        messages = [{"role": "user", "content": "hi"}]
        result = format_chat(messages, add_generation_prompt=False)
        # Should NOT end with assistant prompt
        assert result.count("<|assistant|>") == 0

    def test_format_chat_with_thinking(self):
        messages = [{"role": "user", "content": "explain this"}]
        result = format_chat_with_thinking(messages, enable_thinking=True)
        assert "<|think_start|>" in result

    def test_format_chat_without_thinking(self):
        messages = [{"role": "user", "content": "explain this"}]
        result = format_chat_with_thinking(messages, enable_thinking=False)
        assert "<|think_start|>" not in result
