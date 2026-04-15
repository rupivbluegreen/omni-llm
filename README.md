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
│  └── Trained: Pretrain (+FIM inline) → SFT → DPO       │
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

Full pipeline, end-to-end — the commands below have been validated on an M1 Pro
at a reduced config (67M params, 2k steps) with loss going from random-init
entropy (10.77) to ~5.0 and the gate script correctly blocking a weak
checkpoint from promotion. For the production 200M / 100K-step run, rent a
cloud GPU — see "Where to run" below.

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
echo 'HF_TOKEN=hf_xxx' > .env     # for gated starcoderdata

# 2. Download raw data. --max-tokens now auto-derives a per-dataset cap;
#    without it, streaming sources like starcoderdata can write 100+ GB.
python -m data.download --output-dir data/raw --datasets code \
    --max-tokens 300_000_000

# 3. Decontaminate against HumanEval / MBPP / MT-Bench. Must run BEFORE
#    tokenization or eval numbers are meaningless.
python -m data.decontaminate --input-dir data/processed/shards \
    --output-dir data/clean

# 4. Train tokenizer. The trainer reads raw source files, so extract
#    documents to a single corpus file first:
python -c "import json; [print(json.loads(l).get('text',''))
    for f in ['data/clean/pretrain_mix_shard_00000.jsonl',
              'data/clean/pretrain_mix_shard_00001.jsonl']
    for l in open(f) if l.strip()]" > data/tokenizer_corpus/corpus.txt
python -m tokenizer.train_tokenizer --data-dir data/tokenizer_corpus \
    --vocab-size 32768 --output tokenizer.json

# 5. Emit model/config.json (defaults are the 200M production config)
python -c "from model.config import OmniscientConfig; \
    OmniscientConfig().save('model/config.json')"

# 6. Pretrain. --fim-rate 0.5 folds FIM into pretrain StarCoder-style.
#    --keep-last-n rolling-prunes old checkpoints; step-*-final is preserved.
python -m training.pretrain \
    --config model/config.json --data-dir data/clean \
    --tokenizer tokenizer.json --output-dir checkpoints/pretrain \
    --max-steps 100000 --batch-size 2 --grad-accum 16 \
    --fim-rate 0.5 --keep-last-n 3

# 7. Gate — blocks promotion if PPL/HumanEval/MT-Bench regress below floors.
python -m evals.gate \
    --checkpoint checkpoints/pretrain/step-100000-final \
    --stage pretrain --eval-data data/heldout.jsonl

# 8. SFT → gate → DPO → gate (same pattern)
python -m training.sft  --checkpoint checkpoints/pretrain/step-100000-final \
    --data data/sft.jsonl --max-steps 8000
python -m training.dpo  --checkpoint checkpoints/sft/best \
    --data data/dpo.jsonl
```

### Where to run

- **M1/M2 Pro (16-32GB)** — fine for the 67M smoke config to validate plumbing
  (~1 hour for 2k steps). The production 200M / 100K-step config is not
  feasible on laptop hardware; expect days-to-weeks and memory pressure
  triggering Jetsam kills.
- **RTX 4090 desktop (24GB)** — ~15-20× the M1 Pro throughput; feasible for
  the full production run over a weekend.
- **Cloud A100/H100 (RunPod, Lambda, Vast.ai)** — recommended. Full 100k-step
  pretrain on 13B tokens is ~10-15 hours on a single A100 at roughly
  $30-60 total cost. Cheapest path to a real model.

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

1. **Pretrain** (100K steps) — 60% code + 25% code-adjacent NL + 15% general NL. FIM (PSM/SPM) is applied inline at `--fim-rate 0.5`, StarCoder-style, so the model learns left-to-right and fill-in-the-middle jointly.
2. **Stage gate** — `python -m evals.gate` runs held-out PPL + HumanEval pass@1 + MT-Bench slice and must pass before the next stage.
3. **SFT** (≤8K steps, capped by `--max-steps`) — Multi-turn conversations with assistant-only loss masking.
4. **Stage gate** — same gate script; regression on any metric blocks promotion.
5. **DPO** (3K steps) — Preference alignment with frozen reference model (β=0.1).

All pretrain shards must be run through `python -m data.decontaminate` against HumanEval / MBPP / MT-Bench prompts before tokenization — eval numbers are meaningless otherwise. The decontamination script loads eval sets from HuggingFace (`openai/openai_humaneval`, `HuggingFaceH4/mt_bench_prompts`) and does 13-gram overlap filtering; expect ~14k distinct n-grams and a handful of real leaks removed per 140k-doc shard in a typical code mix.

## Validated end-to-end (M1 Pro, 67M smoke config)

The entire pipeline above has been exercised on an M1 Pro 32GB:

| Stage | Result |
|---|---|
| Env + MLX on GPU | ✅ `Device(gpu, 0)` |
| Data download (300M-token cap, 7 starcoderdata subsets) | ✅ 140k docs, ~195M tokens, ~800 MB raw |
| Decontamination | ✅ 14,551 n-grams, 35 real leaks removed |
| Tokenizer (BPE 32k) | ✅ FIM sentinels at IDs 4/5/6 |
| Pretrain (67M, 2000 steps, seq 1024, batch 2, accum 4, fim-rate 0.5) | ✅ loss 10.77 → ~5.05, 5000 tok/s sustained, ~53 min wall-clock |
| Checkpoint hygiene (`--keep-last-n 3`) | ✅ rolling prune works, `-final` preserved |
| Gate script | ✅ PPL 305, HumanEval 0.0, MT-Bench 0.0 — correctly **FAIL**ed a weak checkpoint, which is the intended behavior |

The 67M / 2k-step run is a plumbing test, not a useful model. To get real quality, run the 200M config for 100K steps on cloud hardware (see "Where to run").
