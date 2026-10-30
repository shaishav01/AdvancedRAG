# Step 3 — adds a temporary /debug/complete endpoint to exercise the LiteLLM gateway (fallback + cache).
# Logfire is configured before other app imports so spans are captured from the start.
import time

import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.gateway import complete, extract_cache_status
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

app = FastAPI(title="AdvancedRAG API")

@app.get("/")
def home():
    return {"message": "AdvancedRAG API is live.", "step": 3}


@app.get("/search")
def search(q: str, limit: int = 15, top_n: int = 5):
    """Temporary demo endpoint: vector top-`limit` vs the reranked top-`top_n`."""
    raw = search_enterprise_knowledge(q, limit=limit)
    reranked_texts = rerank_documents(q, [r["content"] for r in raw], top_n=top_n)
    by_text = {r["content"]: r for r in raw}

    def view(r: dict) -> dict:
        return {"source": r["source"], "vector_score": round(r["score"], 4), "preview": r["content"][:200]}

    return {
        "query": q,
        "vector_hits": [view(r) for r in raw],
        "reranked": [view(by_text[t]) for t in reranked_texts if t in by_text],
    }


class CompleteRequest(BaseModel):
    prompt: str
    temperature: float = 0.1


@app.post("/debug/complete")
def debug_complete(req: CompleteRequest):
    """Temporary demo endpoint: one LLM call through the gateway. Shows which model answered and cache HIT/MISS."""
    start = time.time()
    try:
        resp = complete(req.prompt, temperature=req.temperature)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gateway call failed: {e}")
    return {
        "answer": resp.choices[0].message.content,
        "model": resp.model,
        "cache": extract_cache_status(resp),
        "latency_s": round(time.time() - start, 2),
    }
