from fastapi import Depends, FastAPI

from app.config import Settings, get_settings

app = FastAPI(title="AI Futures Position Recommend Bot", version="0.1.0")


@app.get("/", tags=["system"])
def read_root() -> dict[str, str]:
    """Simple root endpoint for uptime checks."""
    return {"message": "AI Futures Position Recommend Bot is running"}


@app.get("/health", tags=["system"])
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    """Health endpoint that surfaces minimal runtime info."""
    return {"status": "ok", "llm_model": settings.llm_model_name}
