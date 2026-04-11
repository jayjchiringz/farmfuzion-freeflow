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
    allow_origins=["https://farm-fuzion-backend.onrender.com",
                   "http://localhost:3001",
                   "https://kpa-health-ui.onrender.com",
                   "https://kpa-health-api.onrender.com",
    ],
    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
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

# Initialize FreeFlow client
client = None

@app.on_event("startup")
async def startup_event():
    """Initialize FreeFlow client on startup"""
    global client
    try:
        # Your Groq API key is already set in environment
        client = FreeFlowClient()
        print("✅ FreeFlow client initialized successfully")
        print(f"📊 Available providers: {client.list_providers()}")
        
        # Check which providers have keys
        for provider in client.providers:
            num_keys = len(provider.api_keys)
            print(f"  - {provider.name}: {num_keys} API key(s)")
            
    except Exception as e:
        print(f"❌ Failed to initialize FreeFlow client: {e}")
        client = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not client:
        return HealthResponse(
            status="error",
            providers_available=[],
            message="FreeFlow client not initialized"
        )
    
    providers = client.list_providers()
    provider_details = []
    for provider in client.providers:
        provider_details.append(f"{provider.name}: {len(provider.api_keys)} keys")
    
    return HealthResponse(
        status="ok",
        providers_available=providers,
        message=" | ".join(provider_details)
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that uses FreeFlow-LLM"""
    if not client:
        raise HTTPException(status_code=503, detail="FreeFlow client not initialized")
    
    try:
        print(f"📝 Received request with {len(request.messages)} messages")
        
        # Convert messages to dict format
        messages_dict = [msg.dict() for msg in request.messages]
        
        # Call FreeFlow client
        print("🤖 Calling FreeFlow client...")
        response = client.chat(
            messages=messages_dict,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model
        )
        
        print(f"✅ Response received from provider: {response.provider}")
        
        # Convert Usage object to dictionary if it exists
        usage_dict = None
        if hasattr(response, 'usage') and response.usage is not None:
            # If usage is a Usage object, convert to dict
            if hasattr(response.usage, '__dict__'):
                usage_dict = response.usage.__dict__
            else:
                usage_dict = dict(response.usage)
        
        return ChatResponse(
            content=response.content,
            provider=response.provider,
            model=response.model or "default",
            usage=usage_dict  # Now it's a dictionary, not an object
        )
        
    except NoProvidersAvailableError as e:
        print(f"❌ All providers exhausted: {e}")
        raise HTTPException(status_code=429, detail="All AI providers rate limited. Please try again later.")
    except Exception as e:
        print(f"❌ Chat error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers")
async def list_providers():
    """List available providers and their key counts"""
    if not client:
        return {"status": "error", "message": "Client not initialized"}
    
    providers_info = []
    for provider in client.providers:
        providers_info.append({
            "name": provider.name,
            "key_count": len(provider.api_keys),
            "default_model": provider.default_model
        })
    
    return {
        "status": "ok",
        "providers": providers_info
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FREE_FLOW_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
    