"""
MachineMachine Agent Bridge
- /v1/embeddings  : BGE-M3 proxy (OpenAI-compatible)
- /escalate       : Create escalation/request between any agents
- /escalations    : List escalations (filter by to/from/status)
- /escalations/{id} : Get or update a single escalation

Agents: m2, muhlmann, kiedis-po, (any = broadcast)
"""

import os, secrets, httpx, uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Union, Optional

app = FastAPI(title="MachineMachine Agent Bridge")
BGE_URL = os.getenv("BGE_URL", "http://memory-embeddings:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://memory-qdrant:6333")
API_KEY = os.getenv("API_KEY", "")
COLLECTION = "agent_escalations"

security = HTTPBearer()

def verify(creds: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY or not secrets.compare_digest(creds.credentials, API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

# ── Embeddings ────────────────────────────────────────────────────────────────

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

# ── Escalation bus ────────────────────────────────────────────────────────────

class EscalationRequest(BaseModel):
    from_agent: str          # e.g. "muhlmann", "kiedis-po", "m2"
    to_agent: str            # e.g. "m2", "muhlmann", "any"
    question: str
    context: Optional[str] = None
    priority: str = "normal" # low | normal | high

class EscalationUpdate(BaseModel):
    status: Optional[str] = None   # acknowledged | resolved
    answer: Optional[str] = None

async def ensure_collection():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5.0)
        if r.status_code == 404:
            await client.put(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=10.0, json={
                "vectors": {"size": 1, "distance": "Cosine"}
            })

@app.on_event("startup")
async def startup():
    try:
        await ensure_collection()
    except Exception:
        pass  # Retry on first request

@app.post("/escalate", dependencies=[Depends(verify)])
async def create_escalation(req: EscalationRequest):
    await ensure_collection()
    esc_id = str(uuid.uuid4())
    payload = {
        "id": esc_id,
        "from_agent": req.from_agent,
        "to_agent": req.to_agent,
        "question": req.question,
        "context": req.context,
        "priority": req.priority,
        "status": "pending",
        "answer": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_at": None,
    }
    async with httpx.AsyncClient() as client:
        r = await client.put(f"{QDRANT_URL}/collections/{COLLECTION}/points", timeout=10.0, json={
            "points": [{"id": esc_id, "vector": [0.0], "payload": payload}]
        })
        if r.status_code not in (200, 206):
            raise HTTPException(status_code=500, detail=f"Qdrant error: {r.text}")
    return {"id": esc_id, "status": "pending"}

@app.get("/escalations", dependencies=[Depends(verify)])
async def list_escalations(to_agent: Optional[str] = None, from_agent: Optional[str] = None, status: Optional[str] = "pending"):
    await ensure_collection()
    conditions = []
    if status:
        conditions.append({"key": "status", "match": {"value": status}})
    if from_agent:
        conditions.append({"key": "from_agent", "match": {"value": from_agent}})
    if to_agent:
        conditions.append({
            "should": [
                {"key": "to_agent", "match": {"value": to_agent}},
                {"key": "to_agent", "match": {"value": "any"}},
            ]
        })
    body = {"limit": 50, "with_payload": True, "with_vector": False,
            "filter": {"must": conditions} if conditions else {}}
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll", json=body, timeout=10.0)
    points = r.json().get("result", {}).get("points", [])
    return {"escalations": [p["payload"] for p in points], "count": len(points)}

@app.get("/escalations/{esc_id}", dependencies=[Depends(verify)])
async def get_escalation(esc_id: str):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{QDRANT_URL}/collections/{COLLECTION}/points/{esc_id}", timeout=5.0)
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="Not found")
    return r.json().get("result", {}).get("payload", {})

@app.patch("/escalations/{esc_id}", dependencies=[Depends(verify)])
async def update_escalation(esc_id: str, update: EscalationUpdate):
    patch = {}
    if update.status:
        patch["status"] = update.status
    if update.answer:
        patch["answer"] = update.answer
    if update.status == "resolved":
        patch["resolved_at"] = datetime.now(timezone.utc).isoformat()
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/payload",
                              json={"payload": patch, "points": [esc_id]}, timeout=10.0)
    if r.status_code not in (200, 206):
        raise HTTPException(status_code=500, detail=f"Qdrant error: {r.text}")
    return {"id": esc_id, **patch}

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    async with httpx.AsyncClient() as client:
        bge = await client.get(f"{BGE_URL}/health", timeout=5.0)
        qdr = await client.get(f"{QDRANT_URL}/healthz", timeout=5.0)
    return {"status": "ok", "bge": bge.status_code == 200, "qdrant": qdr.status_code == 200}
