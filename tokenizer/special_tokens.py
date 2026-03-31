SPECIAL_TOKENS = {
    # Base control
    "PAD": "<|pad|>",
    "BOS": "<|bos|>",
    "EOS": "<|eos|>",
    "UNK": "<|unk|>",
    # FIM (fill-in-middle)
    "FIM_PREFIX": "<|fim_prefix|>",
    "FIM_MIDDLE": "<|fim_middle|>",
    "FIM_SUFFIX": "<|fim_suffix|>",
    # File context
    "FILE_SEP": "<|file_sep|>",
    "REPO_NAME": "<|repo_name|>",
    # Chat roles
    "IM_START": "<|im_start|>",
    "IM_END": "<|im_end|>",
    "ROLE_SYSTEM": "<|system|>",
    "ROLE_USER": "<|user|>",
    "ROLE_ASSISTANT": "<|assistant|>",
    # Thinking / CoT
    "THINK_START": "<|think_start|>",
    "THINK_END": "<|think_end|>",
    # Tool use
    "TOOL_CALL": "<|tool_call|>",
    "TOOL_RESULT": "<|tool_result|>",
}

SPECIAL_TOKEN_LIST = list(SPECIAL_TOKENS.values())


def get_token_id(tokenizer, name: str) -> int:
    """Get the token ID for a special token by name. Uses token ID comparison (not string)."""
    token_str = SPECIAL_TOKENS[name]
    token_id = tokenizer.token_to_id(token_str)
    if token_id is None:
        raise ValueError(f"Special token {name!r} ({token_str}) not found in tokenizer vocabulary")
    return token_id
