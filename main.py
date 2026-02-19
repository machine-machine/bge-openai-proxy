"""
BGE → OpenAI Embeddings Proxy
Translates OpenAI /v1/embeddings format to HuggingFace TEI format (memory-embeddings service).

Upstream: POST /embed {"inputs": str | list[str]} → [[float, ...], ...]
This proxy: POST /v1/embeddings {"model": "...", "input": str | list[str]} → OpenAI response
"""

import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union

app = FastAPI(title="BGE-OpenAI Proxy", version="1.0.0")

BGE_URL = os.getenv("BGE_URL", "http://memory-embeddings:8000")


class EmbedRequest(BaseModel):
    model: str = "bge-m3"
    input: Union[str, list[str]]
    encoding_format: str = "float"


@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BGE_URL}/embed",
            json={"inputs": texts},
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        vectors = resp.json()  # [[float, ...], ...]
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BGE_URL}/health", timeout=5.0)
    return {"status": "ok", "upstream": r.status_code == 200}
