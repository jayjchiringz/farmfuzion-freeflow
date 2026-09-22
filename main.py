from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

# 🔧 FIX: import config alongside the client
from freeflow_llm import FreeFlowClient, NoProvidersAvailableError, config

# Load environment variables
load_dotenv()

app = FastAPI(title="FreeFlow LLM Service for FarmFuzion")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://farm-fuzion-backend.onrender.com",
        "https://farm-fuzion-frontend-vercel.vercel.app",
        "https://farm-fuzion-frontend-vercel-n6r6f0e5b.vercel.app",
        "http://localhost:3001",
        "http://localhost:5173",
        "https://kpa-health-ui.onrender.com",
        "https://kpa-health-api.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 🔧 FIX: Override FreeFlow's retired model defaults
# ============================================
# FreeFlow ships with model names that were retired in mid-2026.
# Override them here with currently-valid names BEFORE creating the client.
#
# Groq:   llama-3.3-70b-versatile / llama-3.1-8b-instant  → SHUT DOWN Aug 16, 2026
# Gemini: gemini-2.5-flash                                → deprecated for new users
#
# These overrides are applied at import time so every provider uses them.

CURRENT_MODELS = {
    "groq": os.getenv("GROQ_DEFAULT_MODEL", "openai/gpt-oss-120b"),
    "gemini": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3.6-flash"),
}

for provider_name, model_id in CURRENT_MODELS.items():
    if provider_name in config.DEFAULT_MODELS:
        old = config.DEFAULT_MODELS[provider_name]
        config.DEFAULT_MODELS[provider_name] = model_id
        print(f"🔧 Overrode {provider_name} default: {old} → {model_id}")
    else:
        config.DEFAULT_MODELS[provider_name] = model_id
        print(f"🔧 Set {provider_name} default: {model_id}")

print(f"📋 Final DEFAULT_MODELS: {config.DEFAULT_MODELS}")


# ============================================
# Request/Response models
# ============================================
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    model: Optional[str] = None  # 🔧 FIX: only used if explicitly set per request


class ChatResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    providers_available: List[str]
    message: str
    default_models: Optional[Dict[str, str]] = None


# Initialize FreeFlow client
client = None


@app.on_event("startup")
async def startup_event():
    """Initialize FreeFlow client on startup"""
    global client
    print("=" * 60)
    print("🚀 FreeFlow Service starting up...")
    print("=" * 60)

    try:
        client = FreeFlowClient()
        print("✅ FreeFlow client initialized successfully")
        print(f"📊 Available providers: {client.list_providers()}")

        total_keys = 0
        for provider in client.providers:
            num_keys = len(provider.api_keys)
            total_keys += num_keys
            effective_model = config.DEFAULT_MODELS.get(provider.name, "(unknown)")
            print(f"  - {provider.name}: {num_keys} API key(s) · model: {effective_model}")

        print(f"📈 Total API keys loaded: {total_keys}")

        if len(client.providers) < 2:
            print("⚠️  WARNING: Only one provider configured. No cross-provider fallback.")
            print("⚠️  Add GEMINI_API_KEY to the Render environment for resilience.")

    except Exception as e:
        print(f"❌ Failed to initialize FreeFlow client: {e}")
        import traceback
        traceback.print_exc()
        client = None

    print("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not client:
        return HealthResponse(
            status="error",
            providers_available=[],
            message="FreeFlow client not initialized",
            default_models=dict(config.DEFAULT_MODELS),
        )

    providers = client.list_providers()
    provider_details = []
    for provider in client.providers:
        provider_details.append(f"{provider.name}: {len(provider.api_keys)} keys")

    return HealthResponse(
        status="ok",
        providers_available=providers,
        message=" | ".join(provider_details),
        default_models=dict(config.DEFAULT_MODELS),
    )


@app.get("/config")
async def config_info():
    """Diagnostic endpoint — resolved config without exposing secrets."""
    if not client:
        return {
            "status": "error",
            "message": "Client not initialized",
            "env": _env_summary(),
            "default_models": dict(config.DEFAULT_MODELS),
        }

    providers_info = []
    total_keys = 0
    for provider in client.providers:
        key_count = len(provider.api_keys)
        total_keys += key_count
        providers_info.append({
            "name": provider.name,
            "key_count": key_count,
            "effective_model": config.DEFAULT_MODELS.get(provider.name, "(unknown)"),
        })

    return {
        "status": "ok",
        "env": _env_summary(),
        "providers": providers_info,
        "default_models": dict(config.DEFAULT_MODELS),
        "total_keys": total_keys,
    }


def _env_summary() -> Dict[str, Any]:
    """Safe summary of the environment (no secret values)."""
    return {
        "groq_keys_configured": _count_keys("GROQ_API_KEY"),
        "gemini_keys_configured": _count_keys("GEMINI_API_KEY"),
        "groq_default_model": CURRENT_MODELS["groq"],
        "gemini_default_model": CURRENT_MODELS["gemini"],
        "free_flow_port": os.getenv("FREE_FLOW_PORT", "8000"),
        "node_env": os.getenv("NODE_ENV", "not set"),
    }


def _count_keys(env_name: str) -> int:
    """Count keys in an env var, whether JSON array or comma-separated."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return 0
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json
            parsed = json.loads(raw)
            return len(parsed) if isinstance(parsed, list) else 1
        except Exception:
            return 1
    return len([k for k in raw.split(",") if k.strip()])


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that uses FreeFlow-LLM"""
    if not client:
        raise HTTPException(status_code=503, detail="FreeFlow client not initialized")

    try:
        print(f"📝 Received request with {len(request.messages)} messages")

        messages_dict = [msg.dict() for msg in request.messages]

        # 🔧 FIX: Do NOT pass a global model override.
        # Each provider uses its own model from config.DEFAULT_MODELS.
        # Only pass `model` when the caller explicitly requests one
        # (and even then, they'd need a provider-specific name).
        print(f"🤖 Calling FreeFlow client... (default models: {dict(config.DEFAULT_MODELS)})")

        response = client.chat(
            messages=messages_dict,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model,  # None → use per-provider default
        )

        print(f"✅ Response from: {response.provider} · model: {response.model}")

        usage_dict = None
        if hasattr(response, "usage") and response.usage is not None:
            if hasattr(response.usage, "__dict__"):
                usage_dict = response.usage.__dict__
            else:
                usage_dict = dict(response.usage)

        return ChatResponse(
            content=response.content,
            provider=response.provider,
            model=response.model or "default",
            usage=usage_dict,
        )

    except NoProvidersAvailableError as e:
        print(f"❌ All providers exhausted: {e}")
        # 🔧 Raise 503 (service unavailable) not 429 — the cause is
        # usually a bad model name, not rate limiting.
        raise HTTPException(
            status_code=503,
            detail=f"All AI providers failed. Check /config for model names. Detail: {str(e)[:300]}",
        )
    except Exception as e:
        print(f"❌ Chat error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def list_providers():
    """List available providers and their key counts."""
    if not client:
        return {"status": "error", "message": "Client not initialized"}

    providers_info = []
    for provider in client.providers:
        providers_info.append({
            "name": provider.name,
            "key_count": len(provider.api_keys),
            "effective_model": config.DEFAULT_MODELS.get(provider.name, "(unknown)"),
        })

    return {
        "status": "ok",
        "providers": providers_info,
        "default_models": dict(config.DEFAULT_MODELS),
    }


if __name__ == "__main__":
    import importlib

    uvicorn = importlib.import_module("uvicorn")
    port = int(os.getenv("FREE_FLOW_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
    