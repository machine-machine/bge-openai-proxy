# BGE → OpenAI Embeddings Proxy

Bridges the HuggingFace Text Embeddings Inference (TEI) API used by `memory-embeddings` to the OpenAI embeddings API format. This lets OpenClaw's native `memory_search` tool use local BGE-M3 embeddings without an OpenAI key.

## Why this exists

| | BGE-M3 (TEI) | OpenClaw `memory_search` |
|--|--|--|
| Endpoint | `POST /embed` | expects `POST /v1/embeddings` |
| Request | `{"inputs": "text"}` | `{"model": "...", "input": "text"}` |
| Response | `[[float, ...]]` | expects `{"data": [{"embedding": [...]}]}` |

## Deploy on Coolify

### 1. Push to GitHub

```bash
cd ~/.openclaw/workspace/projects/bge-openai-proxy
git init && git add -A
git commit -m "init"
git remote add origin git@github.com:machine-machine/bge-openai-proxy.git
git push -u origin main
```

### 2. Create Coolify Service

- **Project**: AI Tools (or same project as memory services)
- **Type**: Dockerfile (from GitHub repo)
- **Network**: Must be on the same Docker network as `memory-embeddings`
- **Port**: 8001 (internal only — no public domain needed)
- **Env var**: `BGE_URL=http://memory-embeddings:8000`
- **Health check**: `GET /health`

> ⚠️ Do NOT expose publicly via Cloudflare — internal only.

### 3. Verify

```bash
curl http://bge-openai-proxy:8001/health
curl -X POST http://bge-openai-proxy:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": "test query"}'
```

## Configure OpenClaw agents

In `openclaw.json`, under `agents.defaults` or per-agent in `agents.list[*]`:

```json
"memorySearch": {
  "enabled": true,
  "provider": "openai",
  "remote": {
    "baseUrl": "http://bge-openai-proxy:8001",
    "apiKey": "dummy"
  },
  "model": "bge-m3"
}
```

### For MuhlmannBot (different Coolify project / network)

If MuhlmannBot runs in a **different Coolify project**, the Docker networks are isolated.
Two options:

**Option A — Same network** (recommended):
Move MuhlmannBot to the same Coolify project as the memory services,
or add it to the same Docker network manually.

**Option B — Expose via internal subdomain**:
Add a Cloudflare-tunneled domain for the proxy:
- Domain: `http://bge-proxy.machinemachine.ai` (internal, no SSL needed)
- Then set `baseUrl: "http://bge-proxy.machinemachine.ai"` in MuhlmannBot's config

> Remember: `http://` not `https://` — Cloudflare handles TLS termination.

## Notes

- BGE-M3 produces **1024-dim** vectors
- Proxy batches all inputs in a single TEI request (efficient)
- No auth required — keep it internal-only
- Stateless — safe to restart anytime
