from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from freeflow_llm import FreeFlowClient, NoProvidersAvailableError

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
# Configuration — resolved at startup
# ============================================
DEFAULT_MODEL = os.getenv("FREE_FLOW_DEFAULT_MODEL", "").strip() or None

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
    model: Optional[str] = None

class ChatResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    providers_available: List[str]
    message: str
    default_model: Optional[str] = None

# Initialize FreeFlow client
client = None

@app.on_event("startup")
async def startup_event():
    """Initialize FreeFlow client on startup"""
    global client
    print("=" * 60)
    print("🚀 FreeFlow Service starting up...")
    print(f"🔧 FREE_FLOW_DEFAULT_MODEL env var: {DEFAULT_MODEL or '(not set)'}")
    print("=" * 60)

    try:
        client = FreeFlowClient()
        print("✅ FreeFlow client initialized successfully")
        print(f"📊 Available providers: {client.list_providers()}")

        total_keys = 0
        for provider in client.providers:
            num_keys = len(provider.api_keys)
            total_keys += num_keys
            model_hint = getattr(provider, "default_model", "(unknown)")
            print(f"  - {provider.name}: {num_keys} API key(s) · default model: {model_hint}")

        print(f"📈 Total API keys loaded: {total_keys}")

        # Warn if only one provider is configured — no fallback chain
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
            default_model=DEFAULT_MODEL,
        )

    providers = client.list_providers()
    provider_details = []
    for provider in client.providers:
        provider_details.append(f"{provider.name}: {len(provider.api_keys)} keys")

    return HealthResponse(
        status="ok",
        providers_available=providers,
        message=" | ".join(provider_details),
        default_model=DEFAULT_MODEL,
    )


@app.get("/config")
async def config_info():
    """Diagnostic endpoint — shows resolved config without exposing secrets."""
    if not client:
        return {
            "status": "error",
            "message": "Client not initialized",
            "env": _env_summary(),
        }

    providers_info = []
    total_keys = 0
    for provider in client.providers:
        key_count = len(provider.api_keys)
        total_keys += key_count
        info = {
            "name": provider.name,
            "key_count": key_count,
        }
        # Defensive — attribute may not exist on all FreeFlow versions
        if hasattr(provider, "default_model"):
            info["default_model"] = provider.default_model
        if hasattr(provider, "models"):
            info["models"] = provider.models
        providers_info.append(info)

    return {
        "status": "ok",
        "env": _env_summary(),
        "providers": providers_info,
        "total_keys": total_keys,
        "fallback_chain_length": total_keys,
    }


def _env_summary() -> Dict[str, Any]:
    """Return a safe summary of the environment (no secret values)."""
    return {
        "groq_keys_configured": _count_keys("GROQ_API_KEY"),
        "gemini_keys_configured": _count_keys("GEMINI_API_KEY"),
        "free_flow_default_model": DEFAULT_MODEL,
        "free_flow_port": os.getenv("FREE_FLOW_PORT", "8000"),
        "node_env": os.getenv("NODE_ENV", "not set"),
    }


def _count_keys(env_name: str) -> int:
    """Count keys in an env var, whether JSON array or comma-separated."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return 0
    # JSON array format: ["k1","k2"]
    if raw.startswith("[") and raw.endswith("]"):
        try:
            import json
            parsed = json.loads(raw)
            return len(parsed) if isinstance(parsed, list) else 1
        except Exception:
            return 1
    # Comma-separated format
    return len([k for k in raw.split(",") if k.strip()])


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that uses FreeFlow-LLM"""
    if not client:
        raise HTTPException(status_code=503, detail="FreeFlow client not initialized")

    try:
        print(f"📝 Received request with {len(request.messages)} messages")

        messages_dict = [msg.dict() for msg in request.messages]

        # Model resolution priority:
        #   1. Explicit model in request (highest priority)
        #   2. FREE_FLOW_DEFAULT_MODEL env var (fallback)
        #   3. None → let FreeFlow pick per-provider default
        resolved_model = request.model or DEFAULT_MODEL
        if resolved_model:
            print(f"🎯 Using model: {resolved_model}")

        print("🤖 Calling FreeFlow client...")
        response = client.chat(
            messages=messages_dict,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=resolved_model,
        )

        print(f"✅ Response received from provider: {response.provider} · model: {response.model}")

        usage_dict = None
        if hasattr(response, "usage") and response.usage is not None:
            if hasattr(response.usage, "__dict__"):
                usage_dict = response.usage.__dict__
            else:
                usage_dict = dict(response.usage)

        return ChatResponse(
            content=response.content,
            provider=response.provider,
            model=response.model or resolved_model or "default",
            usage=usage_dict,
        )

    except NoProvidersAvailableError as e:
        print(f"❌ All providers exhausted: {e}")
        raise HTTPException(
            status_code=429,
            detail="All AI providers rate limited. Please try again later.",
        )
    except Exception as e:
        print(f"❌ Chat error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers")
async def list_providers():
    """List available providers and their key counts (fixed — was 500ing)."""
    if not client:
        return {"status": "error", "message": "Client not initialized"}

    providers_info = []
    for provider in client.providers:
        info: Dict[str, Any] = {
            "name": provider.name,
            "key_count": len(provider.api_keys),
        }
        # 🔧 FIX: only include default_model if the attribute exists
        if hasattr(provider, "default_model"):
            info["default_model"] = provider.default_model
        providers_info.append(info)

    return {
        "status": "ok",
        "providers": providers_info,
        "free_flow_default_model": DEFAULT_MODEL,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FREE_FLOW_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
    