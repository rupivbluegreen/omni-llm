"""POST /v1/chat/completions — OpenAI-compatible chat endpoint."""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "omniscient-200m"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.95
    stream: bool = False
    # OmniscientLLM extensions
    enable_thinking: bool = False
    conversation_id: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "omniscient-200m"
    choices: list[ChatCompletionChoice]
    usage: Usage = Usage()


async def stream_sse(
    generator, request_id: str, model: str
) -> AsyncGenerator[str, None]:
    """Convert token generator to SSE stream."""
    for chunk in generator:
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"

    # Send final chunk
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completions with multi-turn and thinking support."""
    chat_gen = request.app.state.chat_generator
    if chat_gen is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    memory = request.app.state.memory

    # Handle conversation continuity
    conversation_id = body.conversation_id or str(uuid.uuid4())
    messages = [m.model_dump() for m in body.messages]

    # If continuing a conversation, load history
    if body.conversation_id and memory:
        historical = await memory.get_context(
            conversation_id, token_budget=2500
        )
        if historical:
            # Merge: history + latest user message
            messages = historical + [messages[-1]]

    # Save user message to memory
    if memory and messages:
        user_msgs = [m for m in messages if m["role"] == "user"]
        if user_msgs:
            await memory.add_message(
                conversation_id, "user", user_msgs[-1]["content"]
            )

    # Generate
    generator = chat_gen.generate(
        messages=messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        show_thinking=body.enable_thinking,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if body.stream:
        return StreamingResponse(
            stream_sse(generator, request_id, body.model),
            media_type="text/event-stream",
        )

    # Non-streaming: collect full response
    full_response = "".join(generator)

    # Save assistant response to memory
    if memory:
        await memory.add_message(conversation_id, "assistant", full_response)

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=full_response)
            )
        ],
    )
