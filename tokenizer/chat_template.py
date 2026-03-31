from .special_tokens import SPECIAL_TOKENS


def format_message(role: str, content: str) -> str:
    """Format a single message: <|im_start|><|role|>\ncontent<|im_end|>\n"""
    im_start = SPECIAL_TOKENS["IM_START"]
    im_end = SPECIAL_TOKENS["IM_END"]
    role_token = SPECIAL_TOKENS[f"ROLE_{role.upper()}"]
    return f"{im_start}{role_token}\n{content}{im_end}\n"


def format_chat(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Format a list of {"role": ..., "content": ...} messages into chat template.
    If add_generation_prompt, append assistant turn prefix at the end."""
    formatted = ""
    for msg in messages:
        formatted += format_message(msg["role"], msg["content"])
    if add_generation_prompt:
        im_start = SPECIAL_TOKENS["IM_START"]
        assistant = SPECIAL_TOKENS["ROLE_ASSISTANT"]
        formatted += f"{im_start}{assistant}\n"
    return formatted


def format_chat_with_thinking(messages: list[dict], enable_thinking: bool = True) -> str:
    """Format chat, inserting <|think_start|> after assistant prompt when thinking is enabled."""
    formatted = format_chat(messages, add_generation_prompt=True)
    if enable_thinking:
        formatted += SPECIAL_TOKENS["THINK_START"] + "\n"
    return formatted
