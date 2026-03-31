"""OmniscientLLM API server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .chat import router as chat_router
from .completions import router as completions_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model, tokenizer, create generators. Shutdown: cleanup."""
    # These will be None until a model is actually loaded
    app.state.chat_generator = None
    app.state.fim_generator = None
    app.state.memory = None

    # Try to initialize conversation memory (works without model)
    try:
        from server.agent.memory import ConversationMemory

        memory = ConversationMemory(db_path="conversations.db")
        await memory.initialize()
        app.state.memory = memory
    except Exception as e:
        print(f"Warning: Could not initialize conversation memory: {e}")

    # Try to load model (requires MLX + trained weights)
    model_path = Path("checkpoints/latest")
    tokenizer_path = Path("tokenizer/omniscient-tokenizer.json")

    if model_path.exists() and tokenizer_path.exists():
        try:
            from tokenizer.train_tokenizer import load_tokenizer
            from model.config import OmniscientConfig
            from model.transformer import OmniscientModel
            from server.llm.chat_generator import ChatGenerator
            from server.llm.fim_generator import FIMGenerator
            from training.utils import load_checkpoint
            import mlx.core as mx

            tokenizer = load_tokenizer(str(tokenizer_path))
            config = OmniscientConfig.load(model_path / "config.json")
            model = OmniscientModel(config)
            load_checkpoint(model, model_path)
            mx.eval(model.parameters())

            app.state.chat_generator = ChatGenerator(model, tokenizer, config.max_seq_len)
            app.state.fim_generator = FIMGenerator(model, tokenizer, config.max_seq_len)

            if app.state.memory:
                app.state.memory.tokenizer = tokenizer

            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Server running without model — only health/info endpoints available")
    else:
        print("No model checkpoint found. Server running in API-only mode.")

    yield

    # Shutdown
    if app.state.memory:
        await app.state.memory.close()


app = FastAPI(
    title="OmniscientLLM",
    version="0.1.0",
    description="200M parameter conversational coding chat agent",
    lifespan=lifespan,
)

# CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(completions_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": app.state.chat_generator is not None,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "omniscient-200m",
                "object": "model",
                "owned_by": "omniscient",
            }
        ],
    }
