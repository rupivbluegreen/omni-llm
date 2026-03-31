"""SFT data loading and formatting with assistant-only loss masking."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

# Import from sibling package — tokenizer must be available
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.chat_template import format_message
from tokenizer.special_tokens import SPECIAL_TOKENS


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]  # -100 for masked positions


def tokenize_conversation(
    conversation: list[dict],
    tokenizer,
    max_seq_len: int = 4096,
) -> SFTExample:
    """Tokenize a multi-turn conversation with assistant-only loss masking.

    Args:
        conversation: List of {"role": "system"|"user"|"assistant", "content": "..."}
        tokenizer: HuggingFace tokenizer instance
        max_seq_len: Maximum sequence length

    Returns:
        SFTExample with input_ids and labels where labels=-100 for non-assistant tokens
    """
    all_input_ids = []
    all_labels = []

    for turn in conversation:
        role = turn["role"]
        content = turn["content"]

        # Format this turn using the chat template
        turn_text = format_message(role, content)
        turn_ids = tokenizer.encode(turn_text).ids

        if role == "assistant":
            # For assistant turns: mask the prefix (<|im_start|><|assistant|>\n)
            # but compute loss on the content + <|im_end|>\n
            prefix_text = f'{SPECIAL_TOKENS["IM_START"]}{SPECIAL_TOKENS["ROLE_ASSISTANT"]}\n'
            prefix_ids = tokenizer.encode(prefix_text).ids
            prefix_len = len(prefix_ids)

            turn_labels = [-100] * prefix_len + turn_ids[prefix_len:]
        else:
            # System and user turns: mask entirely
            turn_labels = [-100] * len(turn_ids)

        all_input_ids.extend(turn_ids)
        all_labels.extend(turn_labels)

    # Truncate to max_seq_len
    all_input_ids = all_input_ids[:max_seq_len]
    all_labels = all_labels[:max_seq_len]

    return SFTExample(input_ids=all_input_ids, labels=all_labels)


def load_sft_dataset(path: str | Path) -> list[list[dict]]:
    """Load SFT conversations from a JSONL file.
    Each line is a JSON object with a "messages" key containing the conversation."""
    conversations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            conversations.append(data["messages"])
    return conversations


def create_sft_dataloader(
    conversations: list[list[dict]],
    tokenizer,
    batch_size: int = 2,
    max_seq_len: int = 4096,
    shuffle: bool = True,
):
    """Yield batches of tokenized SFT examples.

    Yields dicts with:
        - input_ids: list of list[int], shape (batch_size, seq_len)
        - labels: list of list[int], shape (batch_size, seq_len)
    """
    import random

    examples = [
        tokenize_conversation(conv, tokenizer, max_seq_len)
        for conv in conversations
    ]

    if shuffle:
        random.shuffle(examples)

    # Pad within batch
    pad_id = tokenizer.token_to_id(SPECIAL_TOKENS["PAD"])
    if pad_id is None:
        pad_id = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        if not batch:
            continue

        max_len = min(max(len(ex.input_ids) for ex in batch), max_seq_len)

        batch_input_ids = []
        batch_labels = []

        for ex in batch:
            pad_len = max_len - len(ex.input_ids)
            batch_input_ids.append(ex.input_ids + [pad_id] * pad_len)
            batch_labels.append(ex.labels + [-100] * pad_len)

        yield {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
        }
