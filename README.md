# OmniscientLLM

A 200M-parameter conversational coding chat agent. Trained with MLX on Apple Silicon, served via FastAPI, used through a VS Code extension.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  VS Code Extension (TypeScript)                         │
│  ├── Chat Panel (webview, SSE streaming)                │
│  ├── Inline Completions (FIM via /v1/completions)       │
│  └── Slash Commands (/explain, /fix, /test, /review...) │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────┐
│  FastAPI Server                                         │
│  ├── /v1/chat/completions  → ChatGenerator              │
│  ├── /v1/completions       → FIMGenerator (if suffix)   │
│  ├── /health, /v1/models                                │
│  └── ConversationMemory (SQLite)                        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  OmniscientModel (200M params, MLX)                     │
│  ├── 16 Transformer blocks, GQA (16 query / 4 KV)      │
│  ├── SwiGLU FFN, RMSNorm, RoPE (theta=500K)            │
│  ├── 4096 context, 32K vocab, weight-tied               │
│  └── Trained: Pretrain → SFT → FIM → DPO               │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Server (Linux/Mac)

```bash
pip install -e ".[dev]"
make serve          # starts FastAPI on :8000
make test           # run tests
```

### Training (Mac M1/M2/M3 with MLX)

```bash
pip install -e ".[train]"
make train-tokenizer
make pretrain
make sft
make dpo
```

### VS Code Extension

```bash
cd extension
npm install
npm run compile
# Install via VS Code: Extensions → Install from VSIX
```

## Project Structure

```
model/          — Architecture (config, layers, transformer)
tokenizer/      — Special tokens, BPE training, chat template
data/           — Download, SFT formatting, CoT generation
training/       — Pretrain, SFT, FIM, DPO scripts (MLX)
server/
  llm/          — ChatGenerator, FIMGenerator, prompts
  agent/        — ConversationMemory (SQLite)
  api/          — FastAPI endpoints (OpenAI-compatible)
extension/      — VS Code extension (TypeScript)
evals/          — HumanEval, MT-Bench evaluation stubs
tests/          — pytest tests
```

## Training Pipeline

1. **Pretrain** (100K steps) — 60% code + 25% code-adjacent NL + 15% general NL
2. **SFT** (8K steps) — Multi-turn conversations with assistant-only loss masking
3. **FIM** (10K steps) — Fill-in-the-middle for inline completions
4. **DPO** (3K steps) — Preference alignment with frozen reference model
