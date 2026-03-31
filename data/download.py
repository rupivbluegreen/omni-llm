"""Data download and preparation script for OmniscientLLM pretraining.

Downloads datasets from HuggingFace and prepares a weighted pretraining mix
with a 60/25/15 split across code, code-adjacent NL, and general NL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRETRAINING_DATASETS = {
    # 60% code — The Stack v2 subsets weighted by language
    "code": {
        "source": "the_stack_v2",
        "weight": 0.60,
        "languages": {
            "python": 0.30,
            "javascript": 0.20,
            "typescript": 0.15,
            "java": 0.10,
            "go": 0.08,
            "rust": 0.07,
            "c": 0.05,
            "cpp": 0.05,
        },
    },
    # 25% code-adjacent natural language
    "code_adjacent_nl": {
        "weight": 0.25,
        "subsets": {
            "stackoverflow": {"weight": 0.10, "source": "stackoverflow"},
            "github_readmes": {"weight": 0.05, "source": "github_readmes"},
            "documentation": {"weight": 0.05, "source": "documentation"},
            "github_issues": {"weight": 0.05, "source": "github_issues"},
        },
    },
    # 15% general natural language
    "general_nl": {
        "weight": 0.15,
        "subsets": {
            "fineweb_edu": {"weight": 0.10, "source": "fineweb_edu"},
            "wikipedia_tech": {"weight": 0.05, "source": "wikipedia_tech"},
        },
    },
}


def download_dataset(name: str, config: dict, output_dir: Path) -> Path:
    """Download a HuggingFace dataset with streaming and write to JSONL.

    Args:
        name: Dataset identifier (e.g. 'the_stack_v2', 'stackoverflow').
        config: Configuration dict with at least a 'source' key and optional
            'subset' or 'language' keys for filtering.
        output_dir: Directory to write the downloaded JSONL file.

    Returns:
        Path to the written JSONL file.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: The 'datasets' package is required. "
            "Install it with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.jsonl"

    source = config.get("source", name)
    subset = config.get("subset")
    language = config.get("language")

    print(f"Downloading dataset '{source}'"
          f"{f' (subset={subset})' if subset else ''}"
          f"{f' (language={language})' if language else ''}...",
          file=sys.stderr)

    try:
        kwargs = {"streaming": True}
        if subset:
            kwargs["name"] = subset
        if language:
            kwargs["data_dir"] = language

        ds = load_dataset(source, split="train", **kwargs)

        count = 0
        with open(output_path, "w") as f:
            for example in ds:
                text = example.get("text") or example.get("content") or ""
                if not text.strip():
                    continue
                record = {"text": text, "source": name}
                if language:
                    record["language"] = language
                f.write(json.dumps(record) + "\n")
                count += 1
                if count % 10_000 == 0:
                    print(f"  ... {count} examples written", file=sys.stderr)

        print(f"Finished '{name}': {count} examples -> {output_path}",
              file=sys.stderr)
        return output_path

    except Exception as e:
        print(
            f"ERROR downloading dataset '{name}': {e}\n"
            f"Make sure you have access to the dataset and are logged in "
            f"with `huggingface-cli login` if needed.",
            file=sys.stderr,
        )
        raise


def shard_jsonl(
    input_path: Path,
    output_dir: Path,
    max_lines_per_shard: int = 100_000,
) -> list[Path]:
    """Split a large JSONL file into smaller shards.

    Args:
        input_path: Path to the input JSONL file.
        output_dir: Directory to write shard files.
        max_lines_per_shard: Maximum number of lines per shard.

    Returns:
        List of paths to the created shard files.
    """
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

    print(f"Sharded {input_path.name}: {line_count} lines -> {shard_idx} shards",
          file=sys.stderr)
    return shard_paths


def prepare_pretraining_mix(
    data_dir: Path,
    output_dir: Path,
    max_tokens: int = 20_000_000_000,
) -> None:
    """Read downloaded datasets, sample according to weights, and write sharded output.

    This interleaves data from different sources according to the weights
    defined in PRETRAINING_DATASETS and writes sharded JSONL files.

    Args:
        data_dir: Directory containing downloaded JSONL files.
        output_dir: Directory to write the interleaved, sharded output.
        max_tokens: Approximate maximum number of tokens in the final mix.
    """
    import random

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all source files and their weights
    sources: list[tuple[Path, float]] = []

    # Code datasets (per-language)
    code_cfg = PRETRAINING_DATASETS["code"]
    for lang, lang_weight in code_cfg["languages"].items():
        path = data_dir / f"the_stack_v2_{lang}.jsonl"
        if path.exists():
            sources.append((path, lang_weight))
        else:
            print(f"WARNING: Missing {path}, skipping", file=sys.stderr)

    # Code-adjacent NL
    for subset_name, subset_cfg in PRETRAINING_DATASETS["code_adjacent_nl"]["subsets"].items():
        path = data_dir / f"{subset_name}.jsonl"
        if path.exists():
            sources.append((path, subset_cfg["weight"]))
        else:
            print(f"WARNING: Missing {path}, skipping", file=sys.stderr)

    # General NL
    for subset_name, subset_cfg in PRETRAINING_DATASETS["general_nl"]["subsets"].items():
        path = data_dir / f"{subset_name}.jsonl"
        if path.exists():
            sources.append((path, subset_cfg["weight"]))
        else:
            print(f"WARNING: Missing {path}, skipping", file=sys.stderr)

    if not sources:
        print("ERROR: No source datasets found in data_dir.", file=sys.stderr)
        sys.exit(1)

    # Normalize weights
    total_weight = sum(w for _, w in sources)
    sources = [(p, w / total_weight) for p, w in sources]

    # Estimate tokens per character (~0.25 tokens per char is a rough average)
    tokens_per_char = 0.25
    max_chars = int(max_tokens / tokens_per_char)

    # Read and sample from each source
    print(f"Preparing pretraining mix (target: ~{max_tokens:,} tokens)...",
          file=sys.stderr)

    mixed_output = output_dir / "pretrain_mix.jsonl"
    total_chars = 0

    with open(mixed_output, "w") as out_f:
        # Round-robin through sources weighted by their ratios
        file_handles = {}
        try:
            for path, weight in sources:
                file_handles[path] = open(path)

            exhausted = set()
            while total_chars < max_chars and len(exhausted) < len(sources):
                for path, weight in sources:
                    if path in exhausted:
                        continue

                    # Number of lines to read proportional to weight
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

    print(f"Mixed output: ~{int(total_chars * tokens_per_char):,} tokens "
          f"-> {mixed_output}", file=sys.stderr)

    # Shard the output
    shard_jsonl(mixed_output, output_dir / "shards")

    # Shuffle shards
    shard_dir = output_dir / "shards"
    shards = sorted(shard_dir.glob("*.jsonl"))
    random.shuffle(shards)

    # Write a manifest
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "total_chars": total_chars,
        "estimated_tokens": int(total_chars * tokens_per_char),
        "max_tokens_target": max_tokens,
        "num_shards": len(shards),
        "shards": [str(s.name) for s in shards],
        "sources": {str(p.name): w for p, w in sources},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {manifest_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare pretraining data for OmniscientLLM"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for downloaded data (default: data/raw)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help='Comma-separated list of dataset names to download, or "all" (default: all)',
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20_000_000_000,
        help="Approximate max tokens for the pretraining mix (default: 20B)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to download
    all_datasets = {}

    # Code datasets (one per language)
    code_cfg = PRETRAINING_DATASETS["code"]
    for lang, lang_weight in code_cfg["languages"].items():
        ds_name = f"the_stack_v2_{lang}"
        all_datasets[ds_name] = {
            "source": code_cfg["source"],
            "language": lang,
            "weight": lang_weight,
        }

    # Code-adjacent NL datasets
    for name, cfg in PRETRAINING_DATASETS["code_adjacent_nl"]["subsets"].items():
        all_datasets[name] = {"source": cfg["source"], "weight": cfg["weight"]}

    # General NL datasets
    for name, cfg in PRETRAINING_DATASETS["general_nl"]["subsets"].items():
        all_datasets[name] = {"source": cfg["source"], "weight": cfg["weight"]}

    if args.datasets == "all":
        to_download = all_datasets
    else:
        requested = [s.strip() for s in args.datasets.split(",")]
        to_download = {}
        for name in requested:
            if name in all_datasets:
                to_download[name] = all_datasets[name]
            else:
                print(f"WARNING: Unknown dataset '{name}', skipping. "
                      f"Available: {', '.join(sorted(all_datasets.keys()))}",
                      file=sys.stderr)

    # Download each dataset
    for name, config in to_download.items():
        try:
            download_dataset(name, config, output_dir)
        except Exception as e:
            print(f"Failed to download '{name}': {e}", file=sys.stderr)
            continue

    # Prepare the pretraining mix
    print("\nPreparing pretraining mix...", file=sys.stderr)
    mix_dir = output_dir.parent / "processed"
    prepare_pretraining_mix(output_dir, mix_dir, max_tokens=args.max_tokens)

    print("\nDone!", file=sys.stderr)


if __name__ == "__main__":
    main()
