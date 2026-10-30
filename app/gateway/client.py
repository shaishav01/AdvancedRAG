"""
LLM gateway — LiteLLM Router running in-process (no proxy server, no hosted control plane).

Provides:
  - model fallback: primary Gemini model -> fallback Gemini model
  - retries with backoff on rate-limit / server errors, and cooldown of failing deployments
  - in-memory response cache (resets on restart; use Redis for anything long-lived)

Free-tier quotas are per project and per model, so the fallback is a *different* model.
Model IDs come from env (see app/config.py) and are placeholders — confirm them in AI Studio.
"""
from litellm import Router
from litellm.types.utils import ModelResponse

from app.config import settings

router = Router(
    model_list=[
        {
            "model_name": "rag-primary",
            "litellm_params": {
                "model": f"gemini/{settings.GEMINI_MODEL_PRIMARY}",
                "api_key": settings.GEMINI_API_KEY,
            },
        },
        {
            "model_name": "rag-fallback",
            "litellm_params": {
                "model": f"gemini/{settings.GEMINI_MODEL_FALLBACK}",
                "api_key": settings.GEMINI_API_KEY,
            },
        },
    ],
    fallbacks=[{"rag-primary": ["rag-fallback"]}],
    num_retries=2,
    allowed_fails=3,
    cooldown_time=30,
    cache_responses=True,
)


def complete(prompt: str, temperature: float = 0.1, **kwargs) -> ModelResponse:
    """Single-turn completion through the gateway. Returns the OpenAI-style response object."""
    resp = router.completion(
        model="rag-primary",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    )
    # Router.completion is typed as ModelResponse | CustomStreamWrapper; streaming isn't supported here.
    if not isinstance(resp, ModelResponse):
        raise TypeError("complete() does not support stream=True")
    return resp


def extract_cache_status(response) -> str:
    """'HIT' if LiteLLM served the response from its cache, else 'MISS'."""
    hidden = getattr(response, "_hidden_params", None) or {}
    return "HIT" if hidden.get("cache_hit") else "MISS"
