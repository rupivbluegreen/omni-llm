"""POST /v1/completions — Code completion endpoint with FIM support."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class CompletionRequest(BaseModel):
    model: str = "omniscient-200m"
    prompt: str = ""
    suffix: str | None = None  # If present, use FIM mode
    max_tokens: int = 256
    temperature: float = 0.2
    stop: list[str] | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str = "omniscient-200m"
    choices: list[CompletionChoice]


@router.post("/v1/completions")
async def completions(request: Request, body: CompletionRequest):
    """Code completion endpoint. Routes to FIM when suffix is provided."""
    fim_gen = request.app.state.fim_generator

    if body.suffix is not None:
        # FIM mode: prefix + suffix → generate middle
        result = fim_gen.complete(
            prefix=body.prompt,
            suffix=body.suffix,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
    else:
        # Plain completion mode: just continue from prompt
        result = fim_gen.complete(
            prefix=body.prompt,
            suffix="",
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=body.model,
        choices=[CompletionChoice(text=result)],
    )
