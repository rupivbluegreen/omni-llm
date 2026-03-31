# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmniscientLLM is a 200M-parameter conversational coding chat agent. The model uses an MLX-based transformer (GQA, SwiGLU, RoPE), trained through four phases: pretrain → SFT → FIM → DPO. Served via FastAPI with an OpenAI-compatible API. Used through a VS Code extension with chat and inline completion.

## Architecture

- **`model/`** — Transformer architecture (MLX, Mac-only). `config.py` is pure Python (no MLX). `layers.py` has RMSNorm, GQA Attention, SwiGLU FFN. `transformer.py` has the full `OmniscientModel`.
- **`tokenizer/`** — 18 special tokens (FIM + chat roles + thinking), BPE training script, chat template formatting (`<|im_start|><|role|>\ncontent<|im_end|>\n`).
- **`data/`** — Pretraining data pipeline (60% code / 25% code-NL / 15% general NL), SFT formatting with assistant-only loss masking (labels=-100), CoT generation.
- **`training/`** — MLX training scripts: pretrain, SFT (masked CE loss), FIM (PSM/SPM transforms), DPO (beta=0.1, frozen reference model).
- **`server/`** — FastAPI server. `llm/` has ChatGenerator (token-ID-based stop/thinking filtering) and FIMGenerator. `agent/` has ConversationMemory (async SQLite with token budgeting). `api/` has OpenAI-compatible endpoints.
- **`extension/`** — VS Code extension (TypeScript). Chat webview, inline completions via FIM, slash commands.
- **`evals/`** — HumanEval and MT-Bench evaluation stubs.

## Key Design Decisions

- Token ID comparison for stop/thinking tokens (not string comparison)
- Actual tokenization for token counting in memory/truncation (not `len//4`)
- No self-summarization in ConversationMemory — simple oldest-first truncation
- Explicit FIM vs chat routing: `/v1/completions` with `suffix` field → FIM; `/v1/chat/completions` → chat
- `model/config.py` has zero MLX imports — testable on any platform

## Commands

```bash
# Server (Linux/Mac)
make serve              # uvicorn on :8000
make test               # pytest tests/
make lint               # ruff check + format check

# Training (Mac with MLX)
make train-tokenizer    # BPE tokenizer training
make pretrain           # pretraining loop
make sft                # supervised fine-tuning
make dpo                # DPO alignment

# Evals
make eval               # run HumanEval + MT-Bench
```

## Testing

```bash
cd /root/ci/omni-llm
python -m pytest tests/ -v
python -m pytest tests/test_config.py                              # single file
python -m pytest tests/test_tokenizer.py::TestChatTemplate::test_format_message  # single test
```

Tests mock MLX dependencies — config, tokenizer, SFT data, and API tests all run on Linux without MLX.

## Dependencies

Managed via `pyproject.toml`. Core: `fastapi`, `uvicorn`, `tokenizers`, `pydantic`, `aiosqlite`. Optional groups: `mlx` (Mac training), `train` (datasets, wandb), `dev` (pytest, ruff, httpx).
