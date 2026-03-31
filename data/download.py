"""Data download and preparation script for OmniscientLLM pretraining.

Downloads datasets from HuggingFace and prepares a weighted pretraining mix
with a 60/25/15 split across code, code-adjacent NL, and general NL.

Requires: pip install datasets huggingface_hub
Login first: huggingface-cli login
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Real HuggingFace dataset IDs and their configurations.
# Each entry: (output_name, hf_dataset_id, hf_kwargs, text_field, weight)
#
# All datasets below are native Parquet on HuggingFace — no trust_remote_code needed.
# Verified working with datasets>=3.0 and streaming=True.
#
# Code datasets use bigcode/starcoderdata (gated, requires accepting ToS):
#   - Parquet format, content field = "content", per-language via data_dir
#   - Alternative: bigcode/the-stack-dedup (v1) also works with same field/dir scheme
#   - NOTE: bigcode/the-stack-v2-dedup only contains SWHIDs, NOT actual code content
#
DATASET_REGISTRY: list[tuple[str, str, dict, str, float]] = [
    # ── 60% Code (StarCoderData, by language) ──
    # Gated: requires `huggingface-cli login` and accepting ToS at
    # https://huggingface.co/datasets/bigcode/starcoderdata
    # Field: "content", data_dir: lowercase language name
    ("code_python", "bigcode/starcoderdata", {"data_dir": "python", "split": "train"}, "content", 0.18),
    ("code_javascript", "bigcode/starcoderdata", {"data_dir": "javascript", "split": "train"}, "content", 0.12),
    ("code_typescript", "bigcode/starcoderdata", {"data_dir": "typescript", "split": "train"}, "content", 0.09),
    ("code_java", "bigcode/starcoderdata", {"data_dir": "java", "split": "train"}, "content", 0.06),
    ("code_go", "bigcode/starcoderdata", {"data_dir": "go", "split": "train"}, "content", 0.048),
    ("code_rust", "bigcode/starcoderdata", {"data_dir": "rust", "split": "train"}, "content", 0.042),
    ("code_c", "bigcode/starcoderdata", {"data_dir": "c", "split": "train"}, "content", 0.03),
    ("code_cpp", "bigcode/starcoderdata", {"data_dir": "c++", "split": "train"}, "content", 0.03),
    # ── 25% Code-adjacent NL ──
    # bigcode/stackoverflow-clean: 10.4M posts, Parquet, not gated, field = "content"
    ("stackoverflow", "bigcode/stackoverflow-clean", {"split": "train"}, "content", 0.10),
    # mikex86/stackoverflow-posts: 58.3M posts, Parquet, not gated, field = "Body"
    # (alternative if more data needed)
    # Markdown from starcoderdata as proxy for READMEs/docs (gated)
    ("github_readmes", "bigcode/starcoderdata", {"data_dir": "markdown", "split": "train"}, "content", 0.05),
    # Restructured-text docs from starcoderdata (gated)
    ("documentation", "bigcode/starcoderdata", {"data_dir": "restructuredtext", "split": "train"}, "content", 0.05),
    # GitHub issues subset of starcoderdata (gated, different columns — load separately)
    ("github_issues", "bigcode/starcoderdata", {"data_dir": "github-issues-filtered-structured", "split": "train"}, "content", 0.05),
    # ── 15% General NL ──
    # HuggingFaceFW/fineweb-edu: Parquet, not gated, field = "text"
    # Valid configs: "sample-10BT", "sample-100BT", "sample-350BT", "default", or CC-MAIN-* dumps
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", {"name": "sample-10BT", "split": "train"}, "text", 0.10),
    # wikimedia/wikipedia: Parquet, not gated, field = "text"
    # Config = "YYYYMMDD.lang", e.g. "20231101.en"
    ("wikipedia", "wikimedia/wikipedia", {"name": "20231101.en", "split": "train"}, "text", 0.05),
]

# Fallback datasets if gated starcoderdata is inaccessible.
# codeparrot/github-code-clean: Parquet, NOT gated, no trust_remote_code needed.
# Field: "code", configs: "{Language}-all" (e.g. "Python-all", "JavaScript-all")
# Limited languages: Python, JavaScript, Java, C, C++, C#, PHP, HTML (no TS/Go/Rust)
FALLBACK_CODE = [
    ("code_python", "codeparrot/github-code-clean", {"name": "Python-all", "split": "train"}, "code", 0.18),
    ("code_javascript", "codeparrot/github-code-clean", {"name": "JavaScript-all", "split": "train"}, "code", 0.12),
    ("code_java", "codeparrot/github-code-clean", {"name": "Java-all", "split": "train"}, "code", 0.09),
    ("code_c", "codeparrot/github-code-clean", {"name": "C-all", "split": "train"}, "code", 0.06),
    ("code_cpp", "codeparrot/github-code-clean", {"name": "C++-all", "split": "train"}, "code", 0.06),
]


def download_dataset(
    name: str,
    hf_id: str,
    hf_kwargs: dict,
    text_field: str,
    output_dir: Path,
    max_examples: int | None = None,
) -> Path:
    """Download a HuggingFace dataset with streaming and write to JSONL."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"

    if output_path.exists():
        # Count existing lines
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing > 0:
            print(f"  Skipping '{name}' — already has {existing:,} examples at {output_path}",
                  file=sys.stderr)
            return output_path

    # Build load_dataset kwargs
    kwargs = {"streaming": True}
    split = hf_kwargs.get("split", "train")

    # Pass through HF-specific kwargs (but not 'split')
    for key in ("name", "data_dir", "data_files", "languages"):
        if key in hf_kwargs:
            kwargs[key] = hf_kwargs[key]

    hf_kwargs_display = {k: v for k, v in kwargs.items() if k != "streaming"}
    print(f"  Downloading '{name}' from {hf_id} {hf_kwargs_display}...", file=sys.stderr)

    try:
        ds = load_dataset(hf_id, split=split, **kwargs)
    except Exception as e:
        print(f"  ERROR loading '{name}': {e}", file=sys.stderr)
        raise

    count = 0
    with open(output_path, "w") as f:
        for example in ds:
            # Try the specified text field, then common alternatives
            text = None
            for field in (text_field, "text", "content", "code", "body"):
                if field in example and example[field]:
                    text = example[field]
                    break

            if not text or not text.strip():
                continue

            # Skip very short texts
            if len(text.strip()) < 50:
                continue

            record = {"text": text, "source": name}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 10_000 == 0:
                print(f"    {count:,} examples...", file=sys.stderr, end="\r")

            if max_examples and count >= max_examples:
                print(f"    Reached max_examples={max_examples:,}", file=sys.stderr)
                break

    print(f"  Done '{name}': {count:,} examples -> {output_path}", file=sys.stderr)
    return output_path


def shard_jsonl(
    input_path: Path,
    output_dir: Path,
    max_lines_per_shard: int = 100_000,
) -> list[Path]:
    """Split a large JSONL file into smaller shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    shard_paths = []
    shard_idx = 0
    line_count = 0
    current_file = None

    try:
        with open(input_path) as f:
            for line in f:
                if line_count % max_lines_per_shard == 0:
                    if current_file is not None:
                        current_file.close()
                    shard_path = output_dir / f"{stem}_shard_{shard_idx:05d}.jsonl"
                    shard_paths.append(shard_path)
                    current_file = open(shard_path, "w")
                    shard_idx += 1

                current_file.write(line)
                line_count += 1
    finally:
        if current_file is not None:
            current_file.close()

    print(f"  Sharded {input_path.name}: {line_count:,} lines -> {shard_idx} shards",
          file=sys.stderr)
    return shard_paths


def prepare_pretraining_mix(
    data_dir: Path,
    output_dir: Path,
    max_tokens: int = 20_000_000_000,
) -> None:
    """Read downloaded datasets, sample according to weights, and write sharded output."""
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all downloaded source files and their weights
    sources: list[tuple[Path, float]] = []
    for name, _, _, _, weight in DATASET_REGISTRY:
        path = data_dir / f"{name}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            sources.append((path, weight))
        else:
            print(f"  WARNING: Missing or empty {path}, skipping", file=sys.stderr)

    if not sources:
        print("ERROR: No source datasets found. Run download first.", file=sys.stderr)
        sys.exit(1)

    # Normalize weights to sum to 1
    total_weight = sum(w for _, w in sources)
    sources = [(p, w / total_weight) for p, w in sources]

    print(f"  Found {len(sources)} datasets, preparing mix "
          f"(target: ~{max_tokens:,} tokens)...", file=sys.stderr)

    tokens_per_char = 0.25
    max_chars = int(max_tokens / tokens_per_char)

    mixed_output = output_dir / "pretrain_mix.jsonl"
    total_chars = 0

    with open(mixed_output, "w") as out_f:
        file_handles = {}
        try:
            for path, weight in sources:
                file_handles[path] = open(path)

            exhausted = set()
            while total_chars < max_chars and len(exhausted) < len(sources):
                for path, weight in sources:
                    if path in exhausted:
                        continue

                    lines_to_read = max(1, int(weight * 100))
                    for _ in range(lines_to_read):
                        line = file_handles[path].readline()
                        if not line:
                            exhausted.add(path)
                            break
                        out_f.write(line)
                        total_chars += len(line)

                        if total_chars >= max_chars:
                            break

                    if total_chars >= max_chars:
                        break
        finally:
            for fh in file_handles.values():
                fh.close()

    estimated_tokens = int(total_chars * tokens_per_char)
    print(f"  Mixed output: ~{estimated_tokens:,} tokens -> {mixed_output}",
          file=sys.stderr)

    # Shard the output
    shard_jsonl(mixed_output, output_dir / "shards")

    # Write manifest
    manifest = {
        "total_chars": total_chars,
        "estimated_tokens": estimated_tokens,
        "max_tokens_target": max_tokens,
        "sources": {str(p.name): w for p, w in sources},
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare pretraining data for OmniscientLLM"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw"),
        help="Output directory for downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help=(
            'Comma-separated dataset names, or "all", or a category: '
            '"code", "code_nl", "general_nl" (default: all)'
        ),
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20_000_000_000,
        help="Approximate max tokens for the pretraining mix (default: 20B)",
    )
    parser.add_argument(
        "--max-examples-per-dataset", type=int, default=None,
        help="Cap examples per dataset (useful for testing the pipeline)",
    )
    parser.add_argument(
        "--skip-mix", action="store_true",
        help="Only download, don't prepare the pretraining mix",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter datasets
    if args.datasets == "all":
        to_download = DATASET_REGISTRY
    elif args.datasets == "code":
        to_download = [d for d in DATASET_REGISTRY if d[0].startswith("code_")]
    elif args.datasets == "code_nl":
        to_download = [d for d in DATASET_REGISTRY if d[0] in
                       ("stackoverflow", "github_readmes", "documentation", "github_issues")]
    elif args.datasets == "general_nl":
        to_download = [d for d in DATASET_REGISTRY if d[0] in ("fineweb_edu", "wikipedia")]
    else:
        requested = {s.strip() for s in args.datasets.split(",")}
        to_download = [d for d in DATASET_REGISTRY if d[0] in requested]
        if not to_download:
            available = [d[0] for d in DATASET_REGISTRY]
            print(f"ERROR: No matching datasets. Available: {', '.join(available)}",
                  file=sys.stderr)
            sys.exit(1)

    # Download
    print(f"Downloading {len(to_download)} datasets to {output_dir}...", file=sys.stderr)
    for name, hf_id, hf_kwargs, text_field, weight in to_download:
        try:
            download_dataset(
                name, hf_id, dict(hf_kwargs), text_field, output_dir,
                max_examples=args.max_examples_per_dataset,
            )
        except Exception as e:
            print(f"  FAILED '{name}': {e}", file=sys.stderr)
            continue

    if not args.skip_mix:
        print("\nPreparing pretraining mix...", file=sys.stderr)
        mix_dir = output_dir.parent / "processed"
        prepare_pretraining_mix(output_dir, mix_dir, max_tokens=args.max_tokens)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
