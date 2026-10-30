# AdvancedRAG

Production-style RAG system built step by step: LangGraph agent, NeMo Guardrails, LiteLLM gateway (Gemini), Qdrant + FlashRank retrieval, Logfire/LangSmith observability, RAGAS evals.

> Work in progress — features land one step at a time. The full README arrives in the final step.

## Run (step 0)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # fill in keys as later steps need them
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/ and http://localhost:8000/docs
