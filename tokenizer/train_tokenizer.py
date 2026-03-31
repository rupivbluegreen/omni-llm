"""Train a BPE tokenizer for OmniscientLLM using the HuggingFace tokenizers library."""

import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .special_tokens import SPECIAL_TOKEN_LIST

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".md", ".txt",
}


def collect_files(data_dir: str) -> list[str]:
    """Recursively collect all supported source files from data_dir."""
    files = []
    for root, _dirs, filenames in os.walk(data_dir):
        for fname in filenames:
            if Path(fname).suffix in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return sorted(files)


def train(data_dir: str, output: str, vocab_size: int) -> Tokenizer:
    """Train a BPE tokenizer on files in data_dir and save to output."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKEN_LIST,
        min_frequency=2,
    )

    files = collect_files(data_dir)
    if not files:
        raise RuntimeError(f"No supported files found in {data_dir}")

    print(f"Training on {len(files)} files from {data_dir} (vocab_size={vocab_size})")
    tokenizer.train(files, trainer)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tokenizer.save(output)
    print(f"Tokenizer saved to {output}")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a pretrained tokenizer from a JSON file."""
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for OmniscientLLM")
    parser.add_argument("--data-dir", required=True, help="Path to directory with training data")
    parser.add_argument("--output", default="tokenizer/omniscient-tokenizer.json", help="Output path for tokenizer JSON")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size")
    args = parser.parse_args()

    train(args.data_dir, args.output, args.vocab_size)
