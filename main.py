import os, secrets, httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Union

app = FastAPI()
BGE_URL = os.getenv("BGE_URL", "http://memory-embeddings:8000")
API_KEY = os.getenv("API_KEY", "")  # required — reject if not set

security = HTTPBearer()

def verify(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY or not secrets.compare_digest(creds.credentials, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

class EmbedRequest(BaseModel):
    model: str = "bge-m3"
    input: Union[str, list[str]]
    encoding_format: str = "float"

@app.post("/v1/embeddings", dependencies=[Depends(verify)])
async def embeddings(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BGE_URL}/embed", json={"inputs": texts}, timeout=30.0)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        vectors = resp.json()
    return {"object": "list", "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)], "model": req.model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}

@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BGE_URL}/health", timeout=5.0)
    return {"status": "ok", "upstream": r.status_code == 200}
