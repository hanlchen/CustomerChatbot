#!/usr/bin/env python3
"""
A scripted stand-in for vLLM's OpenAI-compatible server.

Reproduces the exact response shape vLLM returns from
`POST /v1/chat/completions`, including `tool_calls` with stringified JSON
arguments, `finish_reason: "tool_calls"`, and a `usage` block — so the client
code in `llm_providers.py` is exercised against the real wire format by the
real OpenAI SDK, without a GPU or a model download.

It is a test fixture, not a model: replies are scripted, and it echoes back
what it received so tests can assert on the request the client built.

    python mock_vllm_server.py --port 8300
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="Mock vLLM server")

# Scripted turns, consumed in order. Each entry is either
#   {"text": "..."}                                  -> a final answer
#   {"tools": [{"name": ..., "arguments": {...}}]}   -> a tool-call round
SCRIPT: List[Dict[str, Any]] = []
RECEIVED: List[Dict[str, Any]] = []
MODEL_NAME = "mock/Qwen3-8B"


class ScriptPayload(BaseModel):
    script: List[Dict[str, Any]]
    model: Optional[str] = None


@app.post("/__script")
async def set_script(payload: ScriptPayload):
    """Load the turns this server will play back, and reset the recorder."""
    global SCRIPT, RECEIVED, MODEL_NAME
    SCRIPT = list(payload.script)
    RECEIVED = []
    if payload.model:
        MODEL_NAME = payload.model
    return {"ok": True, "turns": len(SCRIPT)}


@app.get("/__received")
async def get_received():
    """Every request body the client sent, in order."""
    return {"requests": RECEIVED}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "vllm"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    RECEIVED.append(body)

    # vLLM validates against the OpenAI schema, where `tools` must have at
    # least one entry. Accepting an empty array here once hid a bug that broke
    # every turn of the two agents that legitimately have no tools: the mock
    # was more forgiving than the thing it stands in for, which is the one way
    # a fixture can actively mislead you.
    if "tools" in body and not body["tools"]:
        raise HTTPException(
            status_code=400,
            detail="[] is too short - 'tools'",
        )

    step = SCRIPT.pop(0) if SCRIPT else {"text": "(script exhausted)"}
    now = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if "tools" in step:
        tool_calls = []
        for call in step["tools"]:
            tool_calls.append({
                "id": f"chatcmpl-tool-{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    # vLLM returns arguments as a JSON *string*, not an object.
                    "arguments": call.get(
                        "arguments_raw",
                        json.dumps(call.get("arguments", {})),
                    ),
                },
            })
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": step.get("text", "")}
        finish_reason = "stop"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": now,
        "model": body.get("model", MODEL_NAME),
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
            "stop_reason": None,
        }],
        "usage": {
            "prompt_tokens": step.get("prompt_tokens", 120),
            "completion_tokens": step.get("completion_tokens", 25),
            "total_tokens": step.get("prompt_tokens", 120) + step.get("completion_tokens", 25),
        },
        "prompt_logprobs": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
