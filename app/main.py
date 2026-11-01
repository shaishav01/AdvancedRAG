# Step 6 — observability. Logfire is configured with the token BEFORE other app imports so every span is captured;
# instrument_fastapi roots each request's trace so Guardrails -> Planner -> Retrieval -> Rerank -> LLM nest under it.
# LangSmith tracing is switched on in app/config.py when LANGSMITH_API_KEY is set.
import os

import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(
    service_name="advancedrag-api",
    token=os.getenv("LOGFIRE_TOKEN"),
    send_to_logfire="if-token-present",
)

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from app.agents.graph import rag_agent
from app.agents.state import AgentState
from app.guardrails import initialize_rails, guard


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_rails()
    yield


app = FastAPI(title="AdvancedRAG API", lifespan=lifespan)
logfire.instrument_fastapi(app)


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    return {"message": "AdvancedRAG API is live.", "step": 6}


@app.get("/health")
def health():
    """Shows which observability backends are active (booleans only — never the keys)."""
    return {
        "status": "ok",
        "logfire_sending": bool(os.getenv("LOGFIRE_TOKEN")),
        "langsmith_tracing": os.environ.get("LANGCHAIN_TRACING_V2") == "true",
    }


@app.get("/graph")
def get_graph_image():
    """Returns the Mermaid PNG of the agent's workflow (needs internet: rendered via mermaid.ink)."""
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query")
def query(request: QueryRequest):
    """Executes the LangGraph RAG flow with per-thread memory."""
    q = request.q
    thread_id = request.thread_id

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": "",
    }
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
            }

        # Gate 2: LangGraph RAG pipeline (runs synchronously to preserve Logfire context)
        final_output = rag_agent.invoke(initial_state, config=config)
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }
    except Exception as e:
        logfire.error(f"Backend execution failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }
